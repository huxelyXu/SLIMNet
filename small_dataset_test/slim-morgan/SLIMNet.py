import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn.functional as F
from tqdm.notebook import tqdm
import os
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = torch.tensor(data_root).float()
        self.label = torch.tensor(data_label).float()

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels


    def __len__(self):
        return len(self.data)
    
class SLIMNet(nn.Module):
    def __init__(self, output_channel=1):
        super(SLIMNet, self).__init__()
        self.out = output_channel
        self.monomer_1 = nn.Linear(167, 167)
        self.monomer_2 = nn.Linear(167, 167)
        # self.monomer_3 = nn.Linear(128, 64)
        self.monomer_4 = nn.Linear(167, 6)
        
        self.scal_1 = nn.Linear(172, 172)
        self.scal_2 = nn.Linear(172, 172)
        self.scal_3 = nn.Linear(172, 2*self.out)
        nn.init.xavier_uniform_(self.scal_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.scal_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.scal_3.weight, gain=nn.init.calculate_gain('relu'))
        
        
        self.chain_1 = nn.Linear(167, 167)
        self.chain_2 = nn.Linear(167, 167)
        self.chain_3 = nn.Linear(167, self.out)
        nn.init.xavier_uniform_(self.chain_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.chain_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.chain_3.weight, gain=nn.init.calculate_gain('relu'))
        
        self.theta_1 = nn.Linear(173, 173)
        self.theta_2 = nn.Linear(173, 173)
        self.theta_3 = nn.Linear(173, self.out)
        self.dropout = nn.Dropout(0.05)
        
        #self.ac  = nn.ReLU()+1
        # self.bn = torch.nn.BatchNorm2d(128)
        
    def ac(self, x): 
        return F.softplus(x)
    
    def forward(self, x):
        monomer = x[:,:167] # 100, 167
        monomer = self.ac(self.monomer_1(monomer)) # 100, 256
        monomer = self.dropout(monomer)
        monomer = self.ac(self.monomer_2(monomer)) # 100, 128
        #monomer = self.ac(self.monomer_3(monomer)) # 100, 64
        monomer_prop = self.monomer_4(monomer) # 100, 6
        
        alpha_o = monomer # 100, 69
        scal_o = torch.concat((monomer, x[:,174:179]), dim=1) # 100, 69
        chain_order_o = torch.concat((monomer, x[:,174:180]), dim=1) # 100, 70

        
        alpha = self.ac(self.chain_1(alpha_o))# 100, 128
        alpha = self.ac(self.chain_2(alpha))# 100, 32
        alpha = self.dropout(alpha)
        alpha += alpha_o
        alpha = self.ac(self.chain_3(alpha)) + 1e-6 # 100, 1
        
        chain_order = self.ac(self.theta_1(chain_order_o)) # 100, 128
        chain_order = self.ac(self.theta_2(chain_order)) # 100, 32
        chain_order = self.dropout(chain_order)
        chain_order += chain_order_o
        chain_order = self.theta_3(chain_order) # 100, 1
        
        scal = self.ac(self.scal_1(scal_o)) #100, 32
        scal = self.ac(self.scal_2(scal))
        scal = self.dropout(scal)
        scal += scal_o
        scal = self.ac(self.scal_3(scal)) + 1e-6
        
        return monomer_prop, torch.log(scal+1), torch.log(alpha), chain_order  
    
    
def train_SLIMNet(model, epochs, train_loader, test_loader, X_test, y_test, out = 1, lam = 10):
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.7, patience=150, verbose=True)
    train_mse = []
    test_mae = []


    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        test_loss = 0.0
        test_R2 = 0.0
        p = 1
        model.train()
        for i_train, data in enumerate(train_loader, 0):
            
            inputs, labels = data
            optimizer.zero_grad()
            monomer_prop, scal, alpha, chain_order = model(inputs)
            beta = scal[:, :out]
            gamma = scal[:, out:]

            mid = torch.pow(gamma, beta)
            prop_guass = torch.mul(alpha, mid)
            polymer_prop = chain_order + prop_guass
           
        
            loss_1 = criterion(polymer_prop, labels)
            
            loss_2 = criterion(inputs[:, 168:174], monomer_prop)
            loss = lam * loss_1 + loss_2 + torch.pow(chain_order, 2)
            loss = torch.sum(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        for i_test, data in enumerate(test_loader, 0):
            inputs, labels = data
            monomer_prop, scal, alpha, chain_order = model(inputs)
            beta = scal[:, :out]
            gamma = scal[:, out:]

            mid = torch.pow(gamma, beta)
            prop_guass = torch.mul(alpha, mid)
  
            polymer_prop = chain_order + prop_guass

            l1loss = nn.L1Loss()
            loss_1 = l1loss(polymer_prop, labels)
            R2 = mean_absolute_error(polymer_prop.detach().numpy(), labels.detach().numpy())
            # loss_2 = l1loss(inputs[:, 168:174], monomer_prop)
            
            loss = loss_1.item()
            test_loss += loss
            test_R2 += R2
            
        scheduler.step(test_loss/(i_test+1))
        
        if epoch % 5 == 4:
            
            # monomer_prop, scal, alpha, chain_order = model(torch.tensor(X_test).float())
            # polymer_prop = chain_order + torch.mul(alpha, torch.pow(scal[:, 4:], scal[:, :4]))
            # R2 = r2_score(polymer_prop.detach().numpy(), y_test)
        # print(f'{epoch+1},{i+1} train_loss:{running_loss/(i_train+1)}, test_loss:{test_loss/(i_test+1)}')
            mmonomer_prop, scal, alpha, chain_order = model(torch.tensor(X_test).float())
            beta = scal[:, :out]
            gamma = scal[:, out:]
            mid = torch.pow(gamma, beta)
            prop_guass = torch.mul(alpha, mid)
            polymer_prop = chain_order + prop_guass
            y_predict_softnet = polymer_prop.detach().numpy()

            MAE = mean_absolute_error(y_test, y_predict_softnet)
            R2 = r2_score(y_test, y_predict_softnet)
            train_mse.append(running_loss / (i_train + 1))
            test_mae.append((MAE))
            best_valid = min(test_mae)
            print('[%d] train_loss: %.5f, MAE: %.5f, R2: %.5f, Best MAE: %.5f' % (epoch+1, running_loss/(i_train+1), MAE, R2, best_valid))

            
        #print(best_valid)
            if (MAE) == best_valid:
                print('saving checkpoint...')
                checkpoint = {'epoch': epoch, 
                      'model_state_dict': model.state_dict(), 
                      'optimizer_state_dict': optimizer.state_dict(), 
                      'scheduler_state_dict': scheduler.state_dict(), 
                      'best_valid_mae': best_valid
                        }
                torch.save(checkpoint, os.path.join('softnet-test.pt'))
                        
    np.save('train_loss.npy', train_mse)
    np.save('test_loss.npy', test_mae)      
           
    print('Finished Training')

if  __name__ == '__main__':
    out = 1
    model = SLIMNet(out)
    data = pd.read_csv('PI1070.csv')
    data = data.fillna(0)
    maccs = []
    for i in data['smiles']:
        molecule = Chem.MolFromSmiles(i) 
        fingerprints = MACCSkeys.GenMACCSKeys(molecule)
        maccs.append(list(fingerprints.ToBitString()))
    
    X0 = np.array(maccs).astype('int') # (1077, 167)
    X1 = np.array(data['mol_weight_monomer']).reshape(-1,1) #(1077, 1)
    X2 = np.array(data['vdw_volume_monomer']).reshape(-1,1) #(1077, 1)
    X3 = np.array(data['qm_total_energy_monomer']).reshape(-1,1) #(1077, 1)
    X4 = np.array(data['qm_homo_monomer']).reshape(-1,1) #(1077, 1)
    X5 = np.array(data['qm_lumo_monomer']).reshape(-1,1) #(1077, 1)
    X6 = np.array(data['qm_dipole_monomer']).reshape(-1,1) #(1077, 1)
    X7 = np.array(data['qm_polarizability_monomer']).reshape(-1,1) #(1077, 1)
    X8 = np.array(data['DP']).reshape(-1,1) #(1077, 1)  
    X9 = np.array(data['Mn']).reshape(-1,1) #(1077, 1)
    X10 = np.array(data['density']).reshape(-1,1) #(1077, 1)
    X11 = np.array(data['compressibility']).reshape(-1,1) #(1077, 1)
    X12 = np.array(data['static_dielectric_const']).reshape(-1,1) #(1077, 1)
    X13 = np.array(data['nematic_order_parameter']).reshape(-1,1) #(1077, 1)

    y0 = np.array(data['thermal_conductivity']).reshape(-1,1)
    y1 = np.array(data['thermal_diffusivity']).reshape(-1,1) #(1077, 1)
    y2 = np.array(data['dielectric_const_dc']).reshape(-1,1) #(1077, 1)
    y3 = np.array(data['volume_expansion']).reshape(-1,1) #(1077, 1)

    y = np.concatenate((y0, y1, y2, y3), axis=1)
    y = y0
    X = np.concatenate((X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13), axis=1)
    
    scalar = StandardScaler()

    X = scalar.fit_transform(X)
    y = scalar.fit_transform(y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
    train_id = np.load('train_id.npy')
    test_id = np.load('test_id.npy')

    X_train, X_test = X[train_id], X[test_id]
    y_train, y_test = y[train_id], y[test_id]

    train_data, test_data = GetLoader(X_train, y_train), GetLoader(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)
    
    train_SLIMNet(model, 70000, train_loader, test_loader, X_test, y_test, out)