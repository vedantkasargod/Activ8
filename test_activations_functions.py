activation_fn = ExponentialFunction(alpha=0.001)
x = torch.Tensor([-2, -1, 0, 1, 2])
output = activation_fn(x)
print(output)

combined_activation_fn = CombinedActivation([LeakyReLU(),
        ELU(alpha=1.0),
        GELU(),
        PReLU(init=0.25),
        Maxout(input_dim=5, num_units=2),  # Adjust input_dim and num_units based on model
        PiecewiseLinearActivation(a1=1, b1=0, a2=0.5, b2=1, c=0),
        ExponentialFunction(alpha=0.5),
        Softplus()])
y = torch.Tensor([-2, -1, 0, 1, 2])
output_combined = combined_activation_fn(y)
print(output_combined)


#output - 
# tensor([0.0001, 0.0004, 0.0010, 0.0027, 0.0074])
# tensor([[0.9094, 0.8713, 2.2844, 2.4781, 2.5085]]
