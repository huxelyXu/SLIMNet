from typing import List, Union

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from .mpn import MPN
from ..args import TrainArgs
from ..features import BatchMolGraph
from ..nn_utils import get_activation_function, initialize_weights, arr_vtf_reverse_norm


class mlp(torch.nn.Module):
    def __init__(self, input_channels=2, output_channels=1):
        super(mlp, self).__init__()
        self.layer1 = torch.nn.Linear(input_channels, 200)
        self.layer2 = torch.nn.Linear(200, output_channels)
        self.dropout = torch.nn.Dropout(0)
        self.ac = torch.nn.Softplus()
    def forward(self, x):
        x = self.ac(self.layer1(x))
        x = self.dropout(x)
        x = self.ac(self.layer2(x))
        return x


class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs, featurizer: bool = False):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param featurizer: Whether the model should act as a featurizer, i.e., outputting the
                           learned features from the last layer prior to prediction rather than
                           outputting the actual property predictions.
        """
        super(MoleculeModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.outputmode = args.outputmode
        self.featurizer = featurizer

        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes
            
        # modify model to output fit parameters rather than conductivity directly   
        if self.outputmode=='arr':
            self.output_size *= 2
        if self.outputmode == 'slimnet':
            self.output_size *= 4
        if self.outputmode=='vtf':
            self.output_size *= 3

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.create_encoder(args)
        self.create_ffn(args)
        self.mlp = mlp()

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs) -> None:
        """
        Creates the message passing encoder for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: TrainArgs) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * args.number_of_molecules
            if args.use_input_features:
                first_linear_dim += args.features_size

        if args.atom_descriptors == 'descriptor':
            first_linear_dim += args.atom_descriptors_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, self.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def featurize(self,
                  batch: Union[List[str], List[Chem.Mol], BatchMolGraph],
                  features_batch: List[np.ndarray] = None,
                  atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Computes feature vectors of the input by running the model except for the last layer.

        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :return: The feature vectors computed by the :class:`MoleculeModel`.
        """
        return self.ffn[:-1](self.encoder(batch, features_batch, atom_descriptors_batch))

    def forward(self,
                batch: Union[List[str], List[Chem.Mol], BatchMolGraph],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                temps_batch: List[np.ndarray] = None
                ) -> torch.FloatTensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :return: The output of the :class:`MoleculeModel`, which is either property predictions
                 or molecule features if :code:`self.featurizer=True`.
        """
        if self.featurizer:
            return self.featurize(batch, features_batch, atom_descriptors_batch)

        output = self.ffn(self.encoder(batch, features_batch, atom_descriptors_batch))

        if self.outputmode == 'slimnet':
            alpha = output[:, 0]
            beta = output[:, 1]
            gamma = output[:, 2]
            phi_theta = output[:, 3]
            
            temps_batch = torch.FloatTensor(temps_batch).to(phi_theta.device)
            mi = torch.cat((phi_theta.unsqueeze(1), temps_batch.unsqueeze(1)), 1)
            phi_theta = self.mlp(mi)
            output = torch.cat((alpha.unsqueeze(1), beta.unsqueeze(1), gamma.unsqueeze(1), phi_theta), 1)

        
        #inverse transform outputs of model if using arrhenius/vtf fits
        #if self.arr_vtf is not None:
        #    output = arr_vtf_reverse_norm(output, self.arr_vtf)
        
        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
        

        return output
