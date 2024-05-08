# 基于机器学习的城市住区建筑布局设计方法研究

本项目构建了计算性辅助设计平台，提供了辅助生成技术路径的建筑师友好交互界面，用于建筑案例的层次检索、自动生成、可视化及分析(见下图)。

![image](https://github.com/yueyueyaoyao/paper/blob/main/images/home_page.png)

## 项目启动方法
软件环境准备：Xshell，Xmanager
网络环境：北京建筑大学校园网
启动步骤：
1. 启动Xshell和Xmanager，在Xshell里新建会话，主机号为10.100.21.209，端口号为22。根据账号密码连入服务器。
2. 在普通用户下（非root）进入/home/YSXT/ 文件夹：
```Python
cd /home/YSXT/
```
3. 在需要运行项目第一部分（辅助生成）时，运行：
```
./1.start_calculate.sh
```
4. 打开浏览器输入http://10.100.21.209:8080/ 即可访问。

<img src="https://github.com/yueyueyaoyao/paper/blob/main/images/The_assisted_generation.jpg" width=30%>


5. 在需要运行项目第二部分（自动生成）时，运行：
```
./2.start_cal.sh
```
6. 打开浏览器输入http://10.100.21.209:8080/ 即可访问。

![image](https://github.com/yueyueyaoyao/paper/blob/main/images/The_automatic_generation.jpg)

7. 返回Xshell, 按ctrl+c关闭当前任务，避免占用5000端口

   

若参考，请标注参考论文
Pedestrian wind flow prediction using spatial-frequency generative adversarial network[J]. Building Simulation, 2023: 1-16．
