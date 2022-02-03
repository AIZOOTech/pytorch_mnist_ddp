大家都知道，mnist 之于深度学习计算机视觉，就像 hello world 对于各大编程语言，我相信很多朋友看各个深度学习框架，都是看一下训练 mnist 的例子，例如 PyTorch 的 mnist 例子，TensorFlow 的 mnist 例子，Paddle 的 mnist 例子。所以说，mnist 对于深度学习框架，是个很好的管中窥豹的机会。

今天开源一个基于 PyTorch 分布式训练，也就是 DistributedDataParallel，简称 DDP，分布式数据并行。

其实，PyTorch 有两个版本的数据并行接口，一个是 `DataParallel`(简称 DP)，另外一个是上面说的 DDP，两者的区别是：

* `DataParallel` 是单进程多线程的，并且只可以在单机跑，但是 DDP 既可以单机跑也可以多机跑，因为 Python 的 GIL 的原因，DP 通常比 DDP 慢， 且DP 因为要额外的分散输入，收集输出，需要在 0 卡占用额外的显存；
* DDP 支持模型并行，但是 DP 当前不支持；

因此，尽量避免使用 DP，直接换成 DDP，其实 DDP 相比 DP，需要改动的代码并不多。

废话不多数，下面简单介绍一下代码。
为了简洁和规范，我是在 PyTorch 的 mnist 例子上改动的。

```
.
├── data # mnist 数据文件夹，会自动联网下载
├── mnist.py # PyTorch 官方 mnist 例子
└── mnist_ddp.py # 笔者写的支持 DDP 的 mnist 例子
```

其实核心的修改地方很少，核心就下面两行

```
if args.distributed:
    # 分布式的需要用分布式 Sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
else:
    train_sampler = torch.utils.data.RandomSampler(dataset_train)
test_sampler = torch.utils.data.SequentialSampler(dataset_val)

if args.distributed:
    # 核心就这一行，将模型包一层就行了
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
```

运行方法
多卡分布式方式

```
python -m torch.distributed.launch --nproc_per_node=4 mnist_ddp.py --batch-size 200 --epochs 20
```

单卡方式：

```
python mnist_ddp.py --batch-size 200 --epochs 20
# 也可以运行 PyTorch 官方版本
python mnist.py --batch-size 200 --epochs 20
```

速度对比

| 卡数  | 20 epoch 耗时(秒) |
|:--- |:--------------:|
| 4   | 73.6           |
| 2   | 137.1          |
| 1   | 242.3          |

可以看到，多卡相对于单卡，是基本接近先行的加速。

因为代码过于简单，大家自己看一下就明白了，可以对照着官方的 `mnist.py `看一下就明白。



***



欢迎AI圈和科技圈的朋友关注我们的公众号，这是我们分享AI技术和资讯的地方。我们要做的事情是搭建开发者和AI算法和产品需求方的一个桥梁，欢迎有AI算法需求的朋友关注我们，也欢迎有熟练算法和开发经验的工程师添加我们，与我们交流。

![](img/wx.png)

**如果你有任何问题，欢迎关注我们的公众号，通过后台给我留言，或者添加作者元峰的微信AIZOOTech与我联系 ，我会将您拉入AIZOO技术交流群。**
我们的技术交流群二维码，欢迎算法开发者和需求方进群交流，请输入备注，例如`张三丰-浙大-目标检测`或者`张三丰-腾讯-图像分割`

** 扫描元峰的微信号，邀请您入群。**

![](img/author.jpg)
