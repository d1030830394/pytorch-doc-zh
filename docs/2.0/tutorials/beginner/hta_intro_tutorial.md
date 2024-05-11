# Introduction to Holistic Trace Analysis

> 译者：[丁兰子](https://github.com/d1030830394)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/hta_intro_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/hta_intro_tutorial.html>

## 如何使用全面追踪分析（Holistic Trace Analysis, HTA）工具

作者： Anupam Bhatnagar

在本教程中，我们将演示如何使用全面追踪分析（Holistic Trace Analysis, HTA）工具来分析分布式训练任务中的追踪数据.

要开始使用，请按照以下步骤操作。

## 安装 HTA
建议使用 Conda 环境来安装 HTA。要安装 Anaconda，请参阅官方 [Anaconda 文档](https://docs.anaconda.com/free/anaconda/install/index.html)。

 1. 使用 pip 安装 HTA：
    ```py
      pip install HolisticTraceAnalysis
    ```
 2. （可选和推荐）设置 Conda 环境：
    ```py
    
     # create the environment env_name
     conda create -n env_name
    
     # activate the environment
     conda activate env_name
    
     # When you are done, deactivate the environment by running ``conda deactivate``
    
    ```
    
## 开始

启动一个Jupyter笔记本并给`traces.trace_dir`设置位置变量。

```py
from hta.trace_analysis import TraceAnalysis
trace_dir = "/path/to/folder/with/traces"
analyzer = TraceAnalysis(trace_dir=trace_dir)
```

## 时间细分配
为了有效地利用 GPU，了解它们在特殊事务上的时间分配至关重要。它们是否主要被分配了计算、通信、 内存事件，或者处于空闲？时间细分功能提供了详细的分析，分析了在这三个类别中花费的时间。

* 空闲时间 - GPU 处于空闲状态。

* 计算时间 - GPU 用于矩阵乘法或向量运算。

* 非计算时间 - GPU 用于通信或内存事件。

为了实现高训练效率，代码应最大限度地利用计算时间，并最大程度地减少空闲时间和非计算时间。以下函数生成一个 DataFrame，提供每个等级的时间使用情况的详细细分。
```py
 analyzer = TraceAnalysis(trace_dir = "/path/to/trace/folder")
 time_spent_df = analyzer.get_temporal_breakdown()
```
![temporal_breakdown_df.png](..%2F..%2Fimg%2Ftemporal_breakdown_df.png)

在 [get_temporal_breakdown](https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_temporal_breakdown) 函数中当参数`visualize`设置为`true`，它还会生成一个条形图，表示按排名细分。

![temporal_breakdown_plot.png](..%2F..%2Fimg%2Ftemporal_breakdown_plot.png)


## 空闲时间细分配

深入了解 GPU 闲置的时间和 其背后的原因可以帮助指导优化策略。GPU 是 当没有内核运行时，将其视为空闲。我们开发了一种 将空闲时间分为三个不同类别的算法：

* 主机等待：指GPU上的空闲时间，由以下原因引起 CPU 对内核的排队速度不够快，无法使 GPU 得到充分利用。 这些类型的低效率可以通过检查 CPU 来解决 导致速度变慢的操作员，增加了批次 尺寸和应用算子融合。

* 内核等待：这是指与启动相关的短暂开销 GPU 上的连续内核。归因于此类别的空闲时间 可以通过使用 CUDA 图形优化来最小化。

* 其他等待：此类别包括当前无法的空闲时间 由于信息不足而归因。可能的原因包括 使用 CUDA 事件和启动延迟在 CUDA 流之间同步 内核。

主机等待时间可以解释为 GPU 因 GPU 而停止的时间 到 CPU。为了将空闲时间归因于内核等待，我们使用以下方法 启发式：
    `连续内核之间的间隙<阈值`
默认阈值为 30 纳秒，可以使用参数进行配置。默认情况下，空闲时间细分为 仅针对排名 0 计算。为了计算其他等级的细分， 在 [get_idle_time_breakdown](https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_idle_time_breakdown) 函数中使用参数。空闲时间细分可以按如下方式生成：consecutive_kernel_delayranks

```py
 analyzer = TraceAnalysis(trace_dir = "/path/to/trace/folder")
 idle_time_df = analyzer.get_idle_time_breakdown()
```

[idle_time_breakdown_percentage.png](..%2F..%2Fimg%2Fidle_time_breakdown_percentage.png)

该函数返回数据帧的元组。第一个数据帧包含 每个等级的每个流上按类别划分的空闲时间。

![idle_time.png](..%2F..%2Fimg%2Fidle_time.png)

当设置为 时，将生成第二个数据帧。它包含每个流的空闲时间的汇总统计信息 在每个等级上。`show_idle_interval_statsTrue`

![idle_time_summary.png](..%2F..%2Fimg%2Fidle_time_summary.png)

 ```
* 提示

    默认情况下，空闲时间细分显示每个 空闲时间类别。将参数设置为 ， 该函数在 y 轴上以绝对时间呈现。`visualize_pctgFalse`
```    
## 内核细分

内核细分功能分解了每种内核类型所花费的时间， 例如通信 （COMM）、计算 （COMP） 和内存 （MEM），所有 对每个类别中花费的时间比例进行排名和显示。这是 以饼图形式显示每个类别所花费的时间百分比：

![kernel_type_breakdown.png](..%2F..%2Fimg%2Fkernel_type_breakdown.png)

内核细分可以按如下方式计算：
```py
 analyzer = TraceAnalysis(trace_dir = "/path/to/trace/folder")
 kernel_type_metrics_df, kernel_metrics_df = analyzer.get_gpu_kernel_breakdown()
```
函数返回的第一个数据帧包含用于 生成饼图。

### 内核持续时间分布
[get_gpu_kernel_breakdown](https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_gpu_kernel_breakdown)返回的第二个数据帧包含每个内核的持续时间摘要统计信息。特别是，这个 包括 count、min、max、average、standard deviation、sum 和 kernel 类型 对于每个等级上的每个内核。

![kernel_metrics_df.png](..%2F..%2Fimg%2Fkernel_metrics_df.png)

使用此数据，HTA 可以创建许多可视化效果来识别性能 瓶颈。

 1. 每个排名的每种内核类型的顶级内核的饼图。

 2. 每个排名前列的平均持续时间的条形图 内核和每种内核类型。

![pie_charts.png](..%2F..%2Fimg%2Fpie_charts.png)

```
* 提示

 所有图像均使用 plotly 生成。将鼠标悬停在图表上会显示 右上角的模式栏，允许用户缩放、平移、选择和 下载图表。
```
上面的饼图显示了排名前 5 位的计算、通信和内存 内核。为每个排名生成类似的饼图。饼图可以是 配置为使用传递的参数显示前 k 个内核 到get_gpu_kernel_breakdown函数。此外，该参数可用于调整 需要分析。如果两者兼而有之 指定，则优先。`num_kernelsduration_rationum_kernelsduration_rationum_kernels`

![comm_across_ranks.png](..%2F..%2Fimg%2Fcomm_across_ranks.png)

上面的条形图显示了 NCCL AllReduce 内核的平均持续时间 跨越所有行列。黑线表示最短和最长时间 在每个等级上都采取。

```py
 警告

 使用 jupyter-lab 时，将“image_renderer”参数值设置为 “jupyterlab”，否则图形将不会在笔记本中呈现。
```
有关此功能的详细演练，请参阅gpu_kernel_breakdown 存储库的 Examples 文件夹中的笔记本。

## 通信计算重叠
在分布式训练中，大量时间花在沟通上 以及 GPU 之间的同步事件。为了实现高 GPU 效率（例如 TFLOPS/GPU），保持 GPU 超额订阅计算至关重要 内核。换句话说，不应因未解析的数据而阻止 GPU 依赖。一种测量计算被阻止的程度的方法 数据依赖性是计算通信计算重叠的。高等 如果通信事件与计算事件重叠，则会观察到 GPU 效率。 缺乏通信和计算重叠将导致 GPU 处于空闲状态， 导致效率低下。 综上所述，更高的通信计算重叠是可取的。要计算 每个等级的重叠百分比，我们衡量以下比率：
 ``` 
                         （通信时计算时间）/（通信时间）
 ```
通信计算重叠可以按如下方式计算：
```py
 analyzer = TraceAnalysis(trace_dir = "/path/to/trace/folder")
 overlap_df = analyzer.get_comm_comp_overlap()
```
该函数返回包含重叠百分比的数据帧 对于每个等级。

![overlap_df.png](..%2F..%2Fimg%2Foverlap_df.png)

当参数设置为 True 时，[get_comm_comp_overlap](https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_comm_comp_overlap) 函数还会生成一个条形图，表示按排名重叠。`visualize`

![overlap_plot.png](..%2F..%2Fimg%2Foverlap_plot.png)


## 增强计数器
### 内存带宽和队列长度计数器

内存带宽计数器测量复制时使用的内存复制带宽 来自 H2D、D2H 和 D2D 的数据通过内存复制 （memcpy） 和内存集 （memset） 事件。HTA 还会计算每个 CUDA 上的未完成操作数 流。我们将其称为队列长度。当流上的**队列长度** 是 1024 或更大，无法在该流上计划新事件，并且 CPU 将停止，直到 GPU 流上的事件得到处理。

[generate_trace_with_counters API](https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.generate_trace_with_counters) 输出具有内存带宽和队列长度的新跟踪文件 计数器。新的跟踪文件包含指示内存的轨道 `memcpy/memset` 操作使用的带宽和队列长度的跟踪 每个流。默认情况下，这些计数器是使用秩 0 生成的 trace 文件，并且新文件的名称中包含后缀。 用户可以选择使用 API 中的参数为多个排名生成计数器。`_with_countersranksgenerate_trace_with_counters`
```py
 analyzer = TraceAnalysis(trace_dir = "/path/to/trace/folder")
 analyzer.generate_trace_with_counters()
```

使用增强计数器生成的跟踪文件的屏幕截图。

![mem_bandwidth_queue_length.png](..%2F..%2Fimg%2Fmem_bandwidth_queue_length.png)

HTA 还提供内存副本带宽和队列长度的摘要 计数器以及 使用以下 API 的代码：
* get_memory_bw_summary
* get_queue_length_summary
* get_memory_bw_time_series
* get_queue_length_time_series

若要查看摘要和时间序列，请使用：
```py
 # generate summary
 mem_bw_summary = analyzer.get_memory_bw_summary()
 queue_len_summary = analyzer.get_queue_length_summary()

 # get time series
 mem_bw_series = analyzer.get_memory_bw_time_series()
 queue_len_series = analyzer.get_queue_length_series()
```

摘要包含计数、最小值、最大值、平均值、标准差、第 25 次、第 50 次、 和第 75 个百分位。

![queue_length_summary.png](..%2F..%2Fimg%2Fqueue_length_summary.png)

时间序列仅包含值更改时的点。一旦一个值是 观察到的时间序列在下一次更新之前保持不变。记忆 带宽和队列长度时间序列函数返回一个字典，其键 是排名，值是该排名的时间序列。默认情况下， 仅针对排名 0 计算时间序列。

## CUDA Kernel Launch 统计

![cuda_kernel_launch.png](..%2F..%2Fimg%2Fcuda_kernel_launch.png)

对于在 GPU 上启动的每个事件，都有一个相应的调度事件 CPU，例如 、 、 。 这些事件由跟踪中的公共相关 ID 链接 - 见图 以上。此功能计算 CPU 运行时事件的持续时间，即其对应的 GPU 内核和启动延迟，例如，GPU 内核启动和 CPU 运算符结束。内核启动信息可以按如下方式生成：`CudaLaunchKernelCudaMemcpyAsyncCudaMemsetAsync`

```py
 analyzer = TraceAnalysis(trace_dir="/path/to/trace/dir")
 kernel_info_df = analyzer.get_cuda_kernel_launch_stats()
```
下面给出了生成的数据帧的屏幕截图。

![cuda_kernel_launch_stats.png](..%2F..%2Fimg%2Fcuda_kernel_launch_stats.png)

CPU 操作的持续时间、GPU 内核和启动延迟使我们能够找到 以下内容：
* **短 GPU 内核** - 持续时间小于相应 CPU 运行时事件。
* **运行时事件异常值** - 持续时间过长的 CPU 运行时事件。
* **启动延迟异常值** - GPU 内核需要很长时间才能调度。

HTA 为上述三个类别中的每一个生成分布图。

**短 GPU 内核**

通常，CPU 端的启动时间范围为 5-20 微秒。在一些 在这种情况下，GPU 执行时间低于启动时间本身。图表 下面帮助我们了解此类实例在代码中出现的频率。

![short_gpu_kernels.png](..%2F..%2Fimg%2Fshort_gpu_kernels.png)

**运行时事件异常值**

运行时异常值取决于用于对异常值进行分类的截止值，因此 [get_cuda_kernel_launch_stats](https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_cuda_kernel_launch_stats) API提供用于配置值的参数。`runtime_cutoff`

![runtime_outliers.png](..%2F..%2Fimg%2Fruntime_outliers.png)

**发射延迟异常值**

发射延迟异常值取决于用于对异常值进行分类的截止值， 因此，_get_cuda_kernel_launch_stats_ API 提供了配置值的参数。 `launch_delay_cutoff`

![launch_delay_outliers.png](..%2F..%2Fimg%2Flaunch_delay_outliers.png)

## 结论

在本教程中，您学习了如何安装和使用 HTA， 一种性能工具，可用于分析分布式中的瓶颈 培训工作流。了解如何使用 HTA 工具执行跟踪 diff 分析，请参阅[使用整体跟踪分析跟踪差异](https://pytorch.org/tutorials/beginner/hta_trace_diff_tutorial.html)。



1) 无需换行的写法: 

$\sqrt{w^T*w}$

2) 需要换行的写法：

$$
\sqrt{w^T*w}
$$

3. 图片参考(用图片的实际地址就行):

<img src='http://data.apachecn.org/img/logo/logo_green.png' width=20% />

4. **翻译完后请删除上面所有模版内容就行**