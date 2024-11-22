# 基于 Python 的锂电池寿命预测[^1][^2][^3]

- （Remaining Useful Life，RUL）& （End Of Life，EOL）

## 依赖项

> [!WARNING]
>
> 此 Fork 仅在 *Windows* `amd64` 实例上测试通过，不保证其他系统环境和依赖项版本可以运行。

```text
########################################
### requirements.txt
# Recommended environment:
# - Windows with x64 architecture
# - CPython >=3.12.7,<=3.13.0
#
# WARNING:
#   Other minor versions of CPython distribution should be OK.
#   But I haven't tested yet.
numpy>=2.1.3
scipy>=1.14.1
matplotlib>=3.9.2
scikit-learn>=1.5.2
notebook
########################################

########################################
### requirements-cu124.txt
# - NVIDIA hardware with CUDA support
#   = CUDA Driver 12.7 or newer
#   = CUDA Toolkit v12.6 Update 2 or newer
#   = CuDNN Runtime v9.5 or newer
-i https://download.pytorch.org/whl/cu124
torch>=2.5.1
########################################
```

## 预测结果

![预测结果](./figures/prediction_nasa.png)

## 常见问题

1. `build_sequences(text, window_size)` 函数生成的预测数据为什么是序列不是下一个点？

   > 序列 `[1, 2, 3, 4, 5]` ， `build_sequences` 函数生成的 `x = [[1, 2, 3], [2, 3, 4]], y=[[2, 3, 4], [3, 4, 5]]` 的目的有两个：
   >
   > - 用序列预测序列，即 `x = [1, 2, 3]` 预测 `y = [2, 3, 4]` ，`x = [2, 3, 4]` 预测 `y = [3, 4, 5]`
   > - 用序列预测下一个点，即 `x = [1, 2, 3]` 预测 `y = [4]` ，`x = [2, 3, 4]` 预测 `y = [5]`
   >
   > 本次实验中，我采用后者。所以，代码中，我训练的时候最后是取了 `train_y` 的最后一列：
   >
   > ```python
   > y = np.reshape(train_y[:, -1] / Rated_Capacity, (-1, 1)).astype(np.float32)
   > ```

## 版本更新

- 2024年11月23日凌晨，修复 `MPL.ipynb` 中的一个错误

    `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument mat1 in method wrapper_CUDA_addmm)`

- 2024年5月12日，修改部分代码以及添加预测图像

- 2022年2月24日，修改部分变量名字

- 2022年2月6日，解决一个错误
    `Tensor for argument #2 'mat1' is on CPU, but expected it to be on GPU (while checking arguments for addmm)`

- 2021年12月1日， 添加数据读取模块。如果原始数据集无法成功读取，可以直接选择加载我已经提取出来的数据 `NASA.npy`

    ```python
    Battery = np.load('NASA.npy', allow_pickle=True)
    Battery = Battery.item()
    ```

## 联系信息

二创修补：[个人博客](https://blog.dragon1573.wang/)

原作：[主页](http://zhouxiuze.com) | [个人博客](http://snailwish.com) | [个人邮箱](mailto:zhouxiuze@foxmail.com)

## 更多内容

1. [基于 Python 的锂电池寿命预测（NASA 锂电池数据集）](https://snailwish.com/395/)

2. [基于 Python 的 MLP 锂电池寿命预测（NASA 锂电池数据集）](https://snailwish.com/427/)

3. [基于 Python 的锂电池寿命预测（CALCE 马里兰大学锂电池数据集）](https://snailwish.com/437/)

4. [基于 Pytorch 的 RNN、LSTM、GRU 寿命预测（NASA 和 CALCE 锂电池数据集）](https://snailwish.com/497/)

5. [基于 Pytorch 的 Transformer 锂电池寿命预测](https://snailwish.com/555/)

## 参考文献

[下载 PDF](https://github.com/XiuzeZhou/xiuzezhou.github.io/tree/main/pub/Transformer.pdf)

```tex
@article{chen2022transformer,
  title={Transformer network for remaining useful life prediction of lithium-ion batteries},
  author={Chen, Daoquan and Hong, Weicong and Zhou, Xiuze},
  journal={Ieee Access},
  volume={10},
  pages={19621--19628},
  year={2022},
  publisher={IEEE}
}
```

-----

[^1]: 使用了 NASA 锂电池数据集。
[^2]: Remaining Useful Life (RUL) ，剩余使用寿命。
[^3]: End Of Life (EOL) ，产品寿命结束。
