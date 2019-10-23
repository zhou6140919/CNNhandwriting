import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 50
LR = 1e-3
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root = 'mnist',
    train = True,
    transform = torchvision.transforms.ToTensor(), #(0, 255) -> (0, 1)
    download = DOWNLOAD_MNIST,
)


# # plot one example
# print(train_data.train_data.size())                 # (60000, 28, 28)
# print(train_data.train_labels.size())               # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()


train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    shuffle = True,
)


test_data = torchvision.datasets.MNIST(root='./mnist/', train = False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(          #1 * 28 * 28
                in_channels=1,            # in_channels (int) – Number of channels in the input image
                out_channels=16,          # out_channels (int) – Number of channels produced by the convolution
                kernel_size=5,            # 扫描5*5的区域kernel_size (int or tuple) – Size of the convolving kernel
                stride=1,
                padding=2,                #if stride==1, padding=(kernel_size-1)/2
            ),           # -> 16 * 28* 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # ->16 * 14 * 14
            )
        self.conv2 = nn.Sequential(      #16 * 14 * 14
            nn.Conv2d(16, 32, 5, 1, 2),  # ->32 * 14 * 14
            nn.ReLU(),                   #32 * 14 * 14
            nn.MaxPool2d(2),             # ->32 * 7 * 7
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # in_features, out_features, bias=True

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)                # batch 32 * 7 * 7
        x = x.view(x.size(0), -1)        #???????????
        output = self.out(x)
        return output


cnn = CNN()
# print(cnn)       #show architecture of net
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)    # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        batch_x = Variable(x)
        batch_y = Variable(y)

        output = cnn(batch_x)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()        # Clears the gradients of all optimized torch.Tensor s.
        loss.backward()              # This function accumulates gradients in the leaves - you might need to zero them before calling it.
        optimizer.step()             # 更新参数

        if step % 50 == 0:
            test_output= cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)


# print 10 predictions from test data
test_output= cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()   #降维
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')