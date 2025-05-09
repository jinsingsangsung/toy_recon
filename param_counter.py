k1 = 5
k2 = 5
C = 3
P = 64
Q = 64

conv2d_params = C*P*k1*k2 + P
ssmconv2d_params = 4*P + 4*C*P + 2*P
scaled_ssmconv2d_params = 4*Q + 4*C*Q + 2*Q*P + 2*P

print(f"conv2d_params: {conv2d_params}")
print(f"ssmconv2d_params: {ssmconv2d_params}")
print(f"scaled_ssmconv2d_params: {scaled_ssmconv2d_params}")

for q in range(64, 102400):
    scaled_ssmconv2d_params = 4*q + 4*C*q + 2*q*P + 2*P
    if scaled_ssmconv2d_params > conv2d_params:
        print(f"Q: {q}, scaled_ssmconv2d_params: {scaled_ssmconv2d_params}")
        break

for q in range(16, 102400):
    scaled_ssmconv2d_params = 4*q + 4*C*q + 2*q
    if scaled_ssmconv2d_params > conv2d_params:
        print(f"Q: {q}, scaled_ssmconv2d_params: {scaled_ssmconv2d_params}")
        break