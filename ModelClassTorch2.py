from ImportFile import *

pi = math.pi


class Swish(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        # pre-activation
        return x * torch.sigmoid(x)


def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif name in ['celu', 'CeLU']:
        return nn.CELU()
    elif name in ['swish']:
        return Swish()
    else:
        raise ValueError('Unknown activation function')


class Pinns(nn.Module):

    def __init__(self, input_dimension, output_dimension, network_properties, additional_models=None,
                 solid_object=None):
        super(Pinns, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_hidden_layers = int(network_properties["hidden_layers"])
        self.neurons = int(network_properties["neurons"])
        self.lambda_residual = float(network_properties["residual_parameter"])
        self.kernel_regularizer = int(network_properties["kernel_regularizer"])
        self.regularization_param = float(network_properties["regularization_parameter"])
        self.num_epochs = int(network_properties["epochs"])
        self.act_string = str(network_properties["activation"])

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(self.n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        self.solid_object = solid_object
        self.additional_models = additional_models

        self.activation = activation(self.act_string)
        # self.a = nn.Parameter(torch.tensor([1.0]))
        
        self.addmask = True
        
        if self.addmask:
            #self.mu_max = torch.tensor([0.25]).to(Ec.dev)
            #self.input_layer2 = nn.Linear(self.input_dimension, self.neurons)
            self.hidden_layers2 = nn.ModuleList(
                [nn.Linear(self.neurons, self.neurons) for _ in range(self.n_hidden_layers - 2)])
            self.output_layer2 = nn.Linear(self.neurons, self.output_dimension)
            

    def forward(self, x):
        inputs = torch.clone(x)
        x = self.activation(self.input_layer(inputs))
        for l in self.hidden_layers:
            x = self.activation(l(x))
        inputs2 = torch.clone(x)
        x = self.output_layer(x)
        if self.addmask:
            mu_max = inputs[:,6] * (0.5*0.3) + (0.5*0.3)
            mask = torch.logical_and((inputs[:,2] >= -mu_max), (inputs[:,2] <= mu_max)) # -mu_max <= mu <= mu_max
            x[~mask,:] = 0.0
            
            #y = self.activation(self.input_layer2(inputs))
            for l in self.hidden_layers2:
                y = self.activation(l(inputs2))
            y = self.output_layer2(y)
            #y = x + y
            y = y * 0.5*(1 - torch.cos(mu_max.unsqueeze(1)*pi))
            
            return torch.cat([x,y], dim=-1)
        return x
    
class PinnsRes(nn.Module):

    def __init__(self, input_dimension, output_dimension, network_properties, additional_models=None,
                 solid_object=None):
        super(PinnsRes, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_hidden_layers = int(network_properties["hidden_layers"])
        self.neurons = int(network_properties["neurons"])
        self.lambda_residual = float(network_properties["residual_parameter"])
        self.kernel_regularizer = int(network_properties["kernel_regularizer"])
        self.regularization_param = float(network_properties["regularization_parameter"])
        self.num_epochs = int(network_properties["epochs"])
        self.act_string = str(network_properties["activation"])

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [ResidualBlock(self.neurons, self.act_string) for _ in range(self.n_hidden_layers)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        self.solid_object = solid_object
        self.additional_models = additional_models

        self.activation = activation(self.act_string)
        # self.a = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        x = self.input_layer(x)
        for l in self.hidden_layers:
            x = l(x)
        return self.output_layer(self.activation(x))

class ResidualBlock(nn.Module):
    def __init__(self, neurons, act_string):
        super(ResidualBlock, self).__init__()
        self.activation = activation(act_string)
        self.fc1 = nn.Linear(neurons, int(neurons/2))
        self.fc2 = nn.Linear(int(neurons/2), int(neurons/2))
        self.fc3 = nn.Linear(int(neurons/2), neurons)

    def forward(self, x):
        y = self.fc1(self.activation(x))
        y = self.fc2(self.activation(y))
        y = self.fc3(self.activation(y))
        return (x + y)


def fit(model, optimizer_ADAM, optimizer_LBFGS, epoch_ADAM, training_set_class, validation_set_clsss=None, verbose=False, training_ic=False):
    num_epochs = model.num_epochs

    train_losses = list([np.NAN])
    val_losses = list()
    freq = 50

    training_coll = training_set_class.data_coll
    training_boundary = training_set_class.data_boundary
    training_initial_internal = training_set_class.data_initial_internal
    
    fraction_coll = int(training_set_class.batches * training_set_class.n_collocation / training_set_class.n_samples)
    fraction_boundary = int(training_set_class.batches * 2 * training_set_class.n_boundary * training_set_class.space_dimensions / training_set_class.n_samples)
    
    torch.autograd.set_detect_anomaly(True)
    
    model.train()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        if epoch < epoch_ADAM:
            if epoch == 0:
                #print("Using ADAM")
                optimizer = optimizer_ADAM
                #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=np.sqrt(0.1), patience=10, threshold=1e-6, threshold_mode='abs', cooldown=10, min_lr=1e-6)
        else:
            print("Using LBFGS")
            optimizer = optimizer_LBFGS
        if verbose and epoch % freq == 0:
            print("################################ ", epoch, " ################################")
            #current_lr = optimizer.param_groups[0]['lr']
            #print("learning rate, best, patience:", current_lr, scheduler.best, scheduler.num_bad_epochs)

        #print(len(training_boundary))
        #print(len(training_coll))
        #print(len(training_initial_internal))

        if len(training_boundary) != 0 and len(training_initial_internal) != 0:
            for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_), (x_u_train_, u_train_)) in enumerate(
                    zip(training_coll, training_boundary, training_initial_internal)):
                if verbose and epoch % freq == 0:
                    print("Batch Number:", step)

                x_ob = None
                u_ob = None

                if torch.cuda.is_available():
                    x_coll_train_ = x_coll_train_.cuda()
                    x_b_train_ = x_b_train_.cuda()
                    u_b_train_ = u_b_train_.cuda()
                    x_u_train_ = x_u_train_.cuda()
                    u_train_ = u_train_.cuda()

                def closure():
                    optimizer.zero_grad()
                    loss_f = CustomLoss()(model, x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, x_ob, u_ob,
                                          training_set_class, training_ic)
                    loss_f.backward()
                    train_losses[0] = loss_f
                    # print(train_losses[0])
                    return loss_f

                optimizer.step(closure=closure)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()
        elif len(training_boundary) == 0 and len(training_initial_internal) != 0:
            for step, ((x_coll_train_, u_coll_train_), (x_u_train_, u_train_)) in enumerate(zip(training_coll, training_initial_internal)):

                x_ob = None
                u_ob = None

                x_b_train_ = torch.full((4, 1), 0)
                u_b_train_ = torch.full((4, 1), 0)

                if torch.cuda.is_available():
                    x_coll_train_ = x_coll_train_.cuda()
                    x_b_train_ = x_b_train_.cuda()
                    u_b_train_ = u_b_train_.cuda()
                    x_u_train_ = x_u_train_.cuda()
                    u_train_ = u_train_.cuda()

                def closure():
                    optimizer.zero_grad()
                    loss_f = CustomLoss()(model, x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, x_ob, u_ob,
                                          training_set_class, training_ic)
                    loss_f.backward()
                    train_losses[0] = loss_f
                    # print(train_losses[0])
                    return loss_f

                optimizer.step(closure=closure)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()
                    
        elif len(training_boundary) != 0 and len(training_initial_internal) == 0:
            epoch = epoch + 0
            print('Epoch:', epoch)
            if True:
                x_b, y_b = Ec.add_boundary(2 * Ec.space_dimensions * training_set_class.n_boundary, seed=epoch)
                x_coll, y_coll = Ec.add_collocations(training_set_class.n_collocation, seed=epoch) #, extend=((epoch+1)/51))
                training_coll = DataLoader(torch.utils.data.TensorDataset(x_coll, y_coll), batch_size=fraction_coll, shuffle=False)
                training_boundary = DataLoader(torch.utils.data.TensorDataset(x_b, y_b), batch_size=fraction_boundary, shuffle=False)
            for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_)) in enumerate(zip(training_coll, training_boundary)):

                x_ob = None
                u_ob = None

                x_u_train_ = torch.full((0, 1), 0)
                u_train_ = torch.full((0, 1), 0)

                if torch.cuda.is_available():
                    x_coll_train_ = x_coll_train_.cuda()
                    x_b_train_ = x_b_train_.cuda()
                    u_b_train_ = u_b_train_.cuda()
                    x_u_train_ = x_u_train_.cuda()
                    u_train_ = u_train_.cuda()

                def closure():
                    optimizer.zero_grad()
                    loss_f = CustomLoss()(model, x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, x_ob, u_ob,training_set_class, training_ic, epoch=epoch)
                    
                    if loss_f is None:
                        return torch.tensor(float('nan'))

                    loss_f.backward()
                    train_losses[0] = loss_f
                    # print(train_losses[0])
                    return loss_f

                # Create a copy of the model's parameters before calling backward()
                old_params = [param.clone().detach() for param in model.parameters()]

                revert = False
                try:
                    optimizer.step(closure=closure)
                    #scheduler.step(train_losses[0].item())
                except:
                    revert = True
                
                # Check for NaN parameters
                if torch.isnan(list(model.parameters())[-1]).any() or revert==True:
                    # Restore model parameters to the previous state
                    with torch.no_grad():
                        for param, old_param in zip(model.parameters(), old_params):
                            param.copy_(old_param)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()

        if validation_set_clsss is not None:

            N_coll_val = validation_set_clsss.n_collocation
            N_b_val = validation_set_clsss.n_boundary
            validation_set = validation_set_clsss.data_no_batches

            for x_val, y_val in validation_set:
                model.eval()

                x_coll_val = x_val[:N_coll_val, :]
                x_b_val = x_val[N_coll_val:N_coll_val + 2 * validation_set_clsss.space_dimensions * N_b_val, :]
                u_b_val = y_val[N_coll_val:N_coll_val + 2 * validation_set_clsss.space_dimensions * N_b_val]
                x_u_val = x_val[N_coll_val:N_coll_val + 2 * validation_set_clsss.space_dimensions * N_b_val:, :]
                u_val = y_val[N_coll_val:N_coll_val + 2 * validation_set_clsss.space_dimensions * N_b_val:, :]

                if torch.cuda.is_available():
                    x_coll_val = x_coll_val.cuda()
                    x_b_val = x_b_val.cuda()
                    u_b_val = u_b_val.cuda()
                    x_u_val = x_u_val.cuda()
                    u_val = u_val.cuda()

                loss_val = CustomLoss()(model, x_u_val, u_val, x_b_val, u_b_val, x_coll_val, validation_set_clsss)

                if torch.cuda.is_available():
                    del x_coll_val
                    del x_b_val
                    del u_b_val
                    del x_u_val
                    del u_val
                    torch.cuda.empty_cache()

                    # val_losses.append(loss_val)
                if verbose and epoch % 100 == 0:
                    print("Validation Loss:", loss_val)

    history = [train_losses, val_losses] if validation_set_clsss is not None else [train_losses]

    # scheduler.step(train_losses[0])

    return train_losses[0]


class CustomLoss(torch.nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, network, x_u_train, u_train, x_b_train, u_b_train, x_f_train, x_obj, u_obj, dataclass,
                training_ic, computing_error=False, epoch=0):

        lambda_residual = network.lambda_residual
        lambda_reg = network.regularization_param
        order_regularizer = network.kernel_regularizer
        space_dimensions = dataclass.space_dimensions
        BC = dataclass.BC
        solid_object = dataclass.obj

        if x_b_train.shape[0] <= 1:
            space_dimensions = 0

        u_pred_var_list = list()
        u_train_var_list = list()
        for j in range(dataclass.output_dimension):

            # Space dimensions
            # Not used for the radiative project, check at the else part
            if not training_ic and Ec.extrema_values is not None:
                for i in range(space_dimensions):
                    half_len_x_b_train_i = int(x_b_train.shape[0] / (2 * space_dimensions))

                    x_b_train_i = x_b_train[i * int(x_b_train.shape[0] / space_dimensions):(i + 1) * int(
                        x_b_train.shape[0] / space_dimensions), :]
                    u_b_train_i = u_b_train[i * int(x_b_train.shape[0] / space_dimensions):(i + 1) * int(
                        x_b_train.shape[0] / space_dimensions), :]
                    boundary = 0
                    while boundary < 2:

                        x_b_train_i_half = x_b_train_i[
                                           half_len_x_b_train_i * boundary:half_len_x_b_train_i * (boundary + 1), :]
                        u_b_train_i_half = u_b_train_i[
                                           half_len_x_b_train_i * boundary:half_len_x_b_train_i * (boundary + 1), :]

                        if BC[i][boundary][j] == "func":
                            u_pred_var_list.append(network(x_b_train_i_half)[:, j])
                            u_train_var_list.append(u_b_train_i_half[:, j])
                        if BC[i][boundary][j] == "der":
                            x_b_train_i_half.requires_grad = True
                            f_val = network(x_b_train_i_half)[:, j]
                            inputs = torch.ones(x_b_train_i_half.shape[0], )
                            if not computing_error and torch.cuda.is_available():
                                inputs = inputs.cuda()
                            der_f_vals = \
                                torch.autograd.grad(f_val, x_b_train_i_half, grad_outputs=inputs, create_graph=True)[0][:, i]
                            u_pred_var_list.append(der_f_vals)
                            u_train_var_list.append(u_b_train_i_half[:, j])
                        elif BC[i][boundary][j] == "periodic":
                            x_half_1 = x_b_train_i_half
                            x_half_2 = x_b_train_i[
                                       half_len_x_b_train_i * (boundary + 1):half_len_x_b_train_i * (boundary + 2), :]
                            x_half_1.requires_grad = True
                            x_half_2.requires_grad = True
                            inputs = torch.ones(x_half_1.shape[0], )
                            if not computing_error and torch.cuda.is_available():
                                inputs = inputs.cuda()
                            pred_first_half = network(x_half_1)[:, j]
                            pred_second_half = network(x_half_2)[:, j]
                            der_pred_first_half = \
                                torch.autograd.grad(pred_first_half, x_half_1, grad_outputs=inputs, create_graph=True)[
                                    0]
                            der_pred_first_half_i = der_pred_first_half[:, i]
                            der_pred_second_half = \
                                torch.autograd.grad(pred_second_half, x_half_2, grad_outputs=inputs, create_graph=True)[
                                    0]
                            der_pred_second_half_i = der_pred_second_half[:, i]

                            u_pred_var_list.append(pred_second_half)
                            u_train_var_list.append(pred_first_half)

                            u_pred_var_list.append(der_pred_second_half_i)
                            u_train_var_list.append(der_pred_first_half_i)

                            boundary = boundary + 1

                        boundary = boundary + 1
            else:
                u_pred_b, u_train_b = Ec.apply_BC(x_b_train, u_b_train, network)
                u_pred_var_list.append(u_pred_b)
                u_train_var_list.append(u_train_b)

                if Ec.time_dimensions != 0:
                    u_pred_0, u_train_0 = Ec.apply_IC(x_u_train, u_train, network)
                    u_pred_var_list.append(u_pred_0)
                    u_train_var_list.append(u_train_0)

            # Time Dimension
            if x_u_train.shape[0] != 0:
                # This is used to solve the radiative inverse problem
                try:
                    if j == 0:
                        if Ec.assign_g:
                            # print("Assign G")
                            g = Ec.get_G(network, x_u_train[:, :3], Ec.n_quad)
                            u_pred_var_list.append(g)

                        else:
                            # print("Not assign G")
                            u_pred_var_list.append(network(x_u_train)[:, j])
                        u_train_var_list.append(u_train[:, j])
                    if j == 1:
                        # print(x_b_train)
                        phys_coord_b = x_b_train[:, :3]
                        if Ec.average:
                            u_pred_var_list.append(Ec.get_average_inf_q(network, phys_coord_b, 10))
                        else:
                            u_pred_var_list.append(network(x_b_train)[:, j])
                        u_train_var_list.append(Ec.K(phys_coord_b[:, 0], phys_coord_b[:, 1], phys_coord_b[:, 2]))
                        # Compute tikonov regularization

                        x_f_train.requires_grad = True
                        k = network(x_f_train)[:, j].reshape(-1, )
                        lambda_k = 0.01

                        grad_k = torch.autograd.grad(k, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ).to(Ec.dev), create_graph=True)[0]

                        grad_k_x = grad_k[:, 0]
                        grad_k_y = grad_k[:, 1]

                        tik_reg = torch.sqrt(lambda_k * torch.mean(grad_k_x ** 2 + grad_k_y ** 2)).reshape(1, )

                        u_pred_var_list.append(tik_reg)
                        u_train_var_list.append(torch.zeros_like(tik_reg))
                except AttributeError:
                    u_pred_var_list.append(network(x_u_train)[:, j])
                    u_train_var_list.append(u_train[:, j])

            if x_obj is not None and not training_ic:
                if BC[-1][j] == "func":
                    u_pred_var_list.append(network(x_obj)[:, j])
                    u_train_var_list.append(u_obj[:, j])

                if BC[-1][j] == "der":
                    x_obj_grad = x_obj.clone()
                    x_obj_transl = x_obj_grad[(np.arange(0, x_obj_grad.shape[0]) + 1) % (x_obj_grad.shape[0]), :]
                    x_obj_mean = (x_obj_grad + x_obj_transl) / 2
                    x_obj_mean.requires_grad = True
                    f_val = network(x_obj_mean)[:, j]
                    inputs = torch.ones(x_obj_mean.shape[0], )
                    if not computing_error and torch.cuda.is_available():
                        inputs = inputs.cuda()
                    der_f_vals_x = torch.autograd.grad(f_val, x_obj_mean, grad_outputs=inputs, create_graph=True)[0][:,
                                   0]
                    der_f_vals_y = torch.autograd.grad(f_val, x_obj_mean, grad_outputs=inputs, create_graph=True)[0][:,
                                   1]
                    t = (x_obj_grad - x_obj_transl)

                    nx = t[:, 1] / torch.sqrt(t[:, 1] ** 2 + t[:, 0] ** 2)
                    ny = -t[:, 0] / torch.sqrt(t[:, 1] ** 2 + t[:, 0] ** 2)
                    der_n = der_f_vals_x * nx + der_f_vals_y * ny
                    u_pred_var_list.append(der_n)
                    u_train_var_list.append(u_obj[:, j])

        u_pred_tot_vars = torch.cat(u_pred_var_list, 0)
        u_train_tot_vars = torch.cat(u_train_var_list, 0)

        if not computing_error and torch.cuda.is_available():
            u_pred_tot_vars = u_pred_tot_vars.cuda()
            u_train_tot_vars = u_train_tot_vars.cuda()

        #print(torch.sum(torch.isnan(u_pred_tot_vars)))
        if torch.isnan(u_pred_tot_vars).any() or torch.isnan(u_train_tot_vars).any():
            print(torch.sum(torch.isnan(u_pred_tot_vars)))
            print(torch.sum(torch.isnan(u_train_tot_vars)))
        #    #assert not torch.isnan(u_pred_tot_vars).any()
            return None

        loss_vars = (torch.mean(abs(u_pred_tot_vars - u_train_tot_vars) ** 2))
        #print(torch.mean(abs(u_pred_tot_vars)))
        #print(torch.mean(abs(u_train_tot_vars)))
        #loss_vars = (abs(u_pred_tot_vars - u_train_tot_vars) ** 2)
        #if torch.isnan(loss_vars).any():
        #    print('# of NaNs:', torch.sum(torch.isnan(loss_vars)))
        #loss_vars = loss_vars[~torch.isnan(loss_vars)]
        #loss_vars = torch.mean(loss_vars)

        if not training_ic:
            
            res = Ec.compute_res(network, x_f_train, space_dimensions, solid_object, computing_error)
            res_train = torch.tensor(()).new_full(size=(res.shape[0],), fill_value=0.0)
            
            if not computing_error and torch.cuda.is_available():
                res = res.cuda()
                res_train = res_train.cuda()

            if Ec.resGrad:
                loss_res_abs = abs(res[:,0])**2
                loss_resGrad = torch.mean(abs(res[:,1:])**2, dim=-1)
                loss_res = loss_res_abs + (Ec.resGradFactor * loss_resGrad)
            else:
                loss_res = abs(res)**2

            if Ec.causality:
                weights, Ec.epsilon = Ec.compute_weights(x_f_train, loss_res, eps=Ec.epsilon)
                loss_res = torch.sum(weights*loss_res)/torch.sum(weights)
            else:
                loss_res = (torch.mean(loss_res))

            u_pred_var_list.append(res)
            u_train_var_list.append(res_train)

        loss_reg = regularization(network, order_regularizer)
        if not training_ic:
            loss_v = torch.log10(
                loss_vars + lambda_residual * loss_res + lambda_reg * loss_reg)  # + lambda_reg/loss_reg
        else:
            loss_v = torch.log10(loss_vars + lambda_reg * loss_reg)
        if Ec.resGrad:
            print("final loss:", loss_v.detach().cpu().numpy().round(4), " ", torch.log10(loss_vars).detach().cpu().numpy().round(4), " ",
              torch.log10(torch.mean(loss_res_abs)).detach().cpu().numpy().round(4), " ", torch.log10(torch.mean(loss_resGrad)).detach().cpu().numpy().round(4))
        else:
            print("final loss:", loss_v.detach().cpu().numpy().round(4), " ", torch.log10(loss_vars).detach().cpu().numpy().round(4), " ",
              torch.log10(loss_res).detach().cpu().numpy().round(4))
            # print(loss_vars.detach().cpu().numpy().round(5))
        return loss_v


def regularization(model, p):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss = reg_loss + torch.norm(param, p)
    return reg_loss


def init_xavier(model):
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            gain = nn.init.calculate_gain('tanh')
            # gain = 1
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.0)

    model.apply(init_weights)


def compute_error(dataset, trained_model):
    training_coll = dataset.data_coll
    training_boundary = dataset.data_boundary
    training_initial_internal = dataset.data_initial_internal
    error_mean = 0
    n = 0
    if len(training_boundary) != 0 and len(training_initial_internal) != 0:
        for step, ((x_coll, u_coll_train_), (x_b, u_b), (x_u, u)) in enumerate(
                zip(training_coll, training_boundary, training_initial_internal)):
            x_ob = None
            u_ob = None
            loss = CustomLoss()(trained_model, x_u, u, x_b, u_b, x_coll, x_ob, u_ob, dataset, False, True)
            error = torch.sqrt(10 ** loss)
            error_mean = error_mean + error
            n = n + 1
        error_mean = error_mean / n
    if len(training_boundary) != 0 and len(training_initial_internal) == 0:
        for step, ((x_coll, u_coll_train_), (x_b, u_b)) in enumerate(
                zip(training_coll, training_boundary)):
            x_ob = None
            u_ob = None
            x_u = torch.full((0, 1), 0)
            u = torch.full((0, 1), 0)
            loss = CustomLoss()(trained_model, x_u, u, x_b, u_b, x_coll, x_ob, u_ob, dataset, False, True)
            error = torch.sqrt(10 ** loss)
            error_mean = error_mean + error
            n = n + 1
        error_mean = error_mean / n
    if len(training_boundary) == 0 and len(training_initial_internal) != 0:
        for step, ((x_coll, u_coll_train_), (x_u, u)) in enumerate(
                zip(training_coll, training_initial_internal)):
            x_ob = None
            u_ob = None
            x_b = torch.full((0, 1), 0)
            u_b = torch.full((0, 1), 0)

            loss = CustomLoss()(trained_model, x_u, u, x_b, u_b, x_coll, x_ob, u_ob, dataset, False, True)
            error = torch.sqrt(10 ** loss)
            error_mean = error_mean + error
            n = n + 1
        error_mean = error_mean / n
    return error_mean


def compute_error_nocoll(dataset, trained_model):
    training_initial_internal = dataset.data_initial_internal
    error_mean = 0
    n = 0
    for step, (x_u, u) in enumerate(training_initial_internal):
        loss = StandardLoss()(trained_model, x_u, u)
        error = torch.sqrt(10 ** loss)
        error_mean = error_mean + error
        n = n + 1
    error_mean = error_mean / n
    return error_mean


class StandardLoss(torch.nn.Module):
    def __init__(self):
        super(StandardLoss, self).__init__()

    def forward(self, network, x_u_train, u_train):
        loss_reg = regularization(network, 2)
        lambda_reg = network.regularization_param
        u_pred = network(x_u_train)

        x_u_train.requires_grad = True
        u = network(x_u_train).reshape(-1, )
        u_ex = torch.tensor(9.) / torch.cosh(np.sqrt(3.) / 2 * (x_u_train[:, 1] - 3 * x_u_train[:, 0])) ** 2
        grad_u = torch.autograd.grad(u, x_u_train, grad_outputs=torch.ones(x_u_train.shape[0], ), create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x, x_u_train, grad_outputs=torch.ones(x_u_train.shape[0]), create_graph=True)[0][:, 1]
        grad_u_xxx = torch.autograd.grad(grad_u_xx, x_u_train, grad_outputs=torch.ones(x_u_train.shape[0]), create_graph=True)[0][:, 1]
        plt.scatter(x_u_train[:, 1].detach().numpy(), u.detach().numpy(), s=10)
        plt.scatter(x_u_train[:, 1].detach().numpy(), grad_u_x.detach().numpy(), s=10)
        plt.scatter(x_u_train[:, 1].detach().numpy(), grad_u_xx.detach().numpy(), s=10)
        plt.scatter(x_u_train[:, 1].detach().numpy(), grad_u_xxx.detach().numpy(), s=10)

        grad_u = torch.autograd.grad(u_ex, x_u_train, grad_outputs=torch.ones(x_u_train.shape[0], ), create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x, x_u_train, grad_outputs=torch.ones(x_u_train.shape[0]), create_graph=True)[0][:, 1]
        grad_u_xxx = torch.autograd.grad(grad_u_xx, x_u_train, grad_outputs=torch.ones(x_u_train.shape[0]), create_graph=True)[0][:, 1]
        plt.scatter(x_u_train[:, 1].detach().numpy(), u_ex.detach().numpy(), s=10, marker="v")
        plt.scatter(x_u_train[:, 1].detach().numpy(), grad_u_x.detach().numpy(), s=10, marker="v")
        plt.scatter(x_u_train[:, 1].detach().numpy(), grad_u_xx.detach().numpy(), s=10, marker="v")
        plt.scatter(x_u_train[:, 1].detach().numpy(), grad_u_xxx.detach().numpy(), s=10, marker="v")
        plt.pause(0.0000001)
        plt.clf()
        loss = torch.log10(torch.mean((u_train - u_pred) ** 2) + lambda_reg * loss_reg)
        del u_train, u_pred
        print(loss)
        return loss


def StandardFit(model, optimizer_ADAM, optimizer_LBFGS, training_set_class, validation_set_clsss=None, verbose=False):
    num_epochs = model.num_epochs

    train_losses = list([np.nan])
    val_losses = list()
    freq = 4

    model.train()
    training_initial_internal = training_set_class.data_initial_internal
    epoch_LSBGF = num_epochs
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        if epoch < epoch_LSBGF:
            print("Using LSBGF")
            optimizer = optimizer_LBFGS
        else:
            print("Using Adam")
            optimizer = optimizer_ADAM
        if verbose and epoch % freq == 0:
            print("################################ ", epoch, " ################################")

        for step, (x_u_train_, u_train_) in enumerate(training_initial_internal):
            if verbose and epoch % freq == 0:
                print("Batch Number:", step)

            if torch.cuda.is_available():
                x_u_train_ = x_u_train_.cuda()
                u_train_ = u_train_.cuda()

            def closure():
                optimizer.zero_grad()
                loss_f = StandardLoss()(model, x_u_train_, u_train_)
                loss_f.backward()
                train_losses[0] = loss_f
                return loss_f

            optimizer.step(closure=closure)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del x_u_train_
            del u_train_

    history = [train_losses, val_losses] if validation_set_clsss is not None else [train_losses]

    return train_losses[0]
