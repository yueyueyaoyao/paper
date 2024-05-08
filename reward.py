import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import copy
import math
import sys
import random
import matplotlib.path as mpltPath
import time
from shapely.geometry import Polygon


#当前是第几个建筑
a=0
#每隔多少帧切换
fps = 3
#总帧数是多少
fps_sum = 21
#每次到10帧计算一次
fream_reward = 0
#每10帧切换一下(0->5)
count = 0

# 地块内初始建筑物坐标
l_real = [[0.53, 0.3], [0, 0.3],  [-0.7, -0.0], [-0.7, -0.5], [0, -0.8], [0.53, -0.6],[0.53, -0.2]]


#周边建筑物坐标（上、下，右）
no_l = [[-0.02, 1.12], [0.82, -1.18], [0.36, -1.18], [-0.10, -1.18], [-0.62, -1.18], [1.12, 1.05], [1.12, 0.55], [1.12, 0.12],
        [1.12, -0.35]]


# 需要排布的建筑物高度
high_arrange = [30, 30, 30, 30, 30, 30, 30]
#周边建筑物的高度
high = [15, 48, 48, 48, 48, 39, 39, 39, 39]

#方位角
As = [-55.3263,-44.0479,-30.9969,-16.1139,0,16.1139,30.9969,44.0479,55.3263]
#影长
Shadow = [6.8572,3.2471,2.2283,1.8187,1.7112,1.8187,2.2283,3.2471,6.8572]

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # 添加智能体
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.blind = True
            agent.length = 0.213
            agent.width = 0.22
            # agent.length = 0.001
            # agent.width = 0.001

        # 添加地标
        world.landmarks = [Landmark() for i in range(7)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.length = 0.213
            landmark.width = 0.22
            # landmark.width = 0.001
            # landmark.length = 0.001

        # =======================no move
        world.landmarks.append(Landmark())
        world.landmarks[7].width = 0.067*2
        world.landmarks[7].length = 0.4
        world.landmarks[7].name = 'landmark %d'
        world.landmarks[7].collide = False
        world.landmarks[7].movable = False

        world.landmarks.append(Landmark())
        world.landmarks[8].width = 0.067*2
        world.landmarks[8].length = 0.1
        world.landmarks[8].name = 'landmark %d'
        world.landmarks[8].collide = False
        world.landmarks[8].movable = False

        world.landmarks.append(Landmark())
        world.landmarks[9].width = 0.067*2
        world.landmarks[9].length = 0.1
        world.landmarks[9].name = 'landmark %d'
        world.landmarks[9].collide = False
        world.landmarks[9].movable = False

        world.landmarks.append(Landmark())
        world.landmarks[10].width = 0.067*2
        world.landmarks[10].length = 0.1
        world.landmarks[10].name = 'landmark %d'
        world.landmarks[10].collide = False
        world.landmarks[10].movable = False

        world.landmarks.append(Landmark())
        world.landmarks[11].width = 0.067*2
        world.landmarks[11].length = 0.1
        world.landmarks[11].name = 'landmark %d'
        world.landmarks[11].collide = False
        world.landmarks[11].movable = False

        world.landmarks.append(Landmark())
        world.landmarks[12].width = 0.067*2
        world.landmarks[12].length = 0.133
        world.landmarks[12].name = 'landmark %d'

        world.landmarks.append(Landmark())
        world.landmarks[13].width = 0.067*2
        world.landmarks[13].length = 0.133
        world.landmarks[13].name = 'landmark %d'

        world.landmarks.append(Landmark())
        world.landmarks[14].width = 0.067*2
        world.landmarks[14].length = 0.133
        world.landmarks[14].name = 'landmark %d'

        world.landmarks.append(Landmark())
        world.landmarks[15].width = 0.067*2
        world.landmarks[15].length = 0.133
        world.landmarks[15].name = 'landmark %d'


        # 制定初始条件
        self.reset_world(world,-99)
        return world

    def reset_world(self, world,num):
        # 给代理的随机属性
        global a
        global reward_num
        global fps
        global l

        l = [[0.53, 0.3], [0, 0.3],  [-0.7, -0.0], [-0.7, -0.5], [0, -0.8], [0.53, -0.6],[0.53, -0.2]]

        reward_num = num
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.9,0.6,0.1])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i < 7:
                landmark.color = landmark.color = np.array([0.75,0.75,0.75])
            if i >= 7:
                landmark.color = landmark.color = np.array([0, 0, 0])

        j = 0
        # 先搞可移动的
        for i, landmark in enumerate(world.landmarks):
            if i < len(l):
                landmark.state.p_pos = np.array(l_real[i])
                i+=1
                landmark.state.p_vel = np.zeros(world.dim_p)
            else:
                landmark.state.p_pos = np.array(no_l[j])
                j += 1
                landmark.state.p_vel = np.zeros(world.dim_p)

        for agent in world.agents:
            agent.state.p_pos = np.array(l_real[a])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def reward(self, agent, world):
        global a
        global count
        global fream_reward
        count = count + 1

        #日照约束
        build_inside = 0
        #退让边界
        bound_res = 0
        #防火间距
        fire = 0
        #绿地边界
        bound_green = 0

        #计算建筑物1的四个顶点
        cur_leftUp = world.agents[0].state.p_pos + [-world.agents[0].length, world.agents[0].width]
        cur_leftDown = world.agents[0].state.p_pos + [-world.agents[0].length, 0]
        cur_rightUp = world.agents[0].state.p_pos + [world.agents[0].length, world.agents[0].width]
        cur_rightDown = world.agents[0].state.p_pos + [world.agents[0].length, 0]

        # 当前建筑的四个点列表
        cur_l = [cur_leftUp, cur_leftDown, cur_rightUp, cur_rightDown]

        #计算地块内其他建筑物的顶点  # 用于判断防火禁入区域
        newl = np.array(copy.deepcopy(l))
        newl = np.delete(newl, a, axis=0)
        other_leftUpArray = newl + np.array([-world.agents[0].length, world.agents[0].width])
        other_leftDownArray = newl + np.array([-world.agents[0].length, 0])
        other_rightUpArray = newl + np.array([world.agents[0].length, world.agents[0].width])
        other_rightDownArray = newl + np.array([world.agents[0].length, 0])
        other_arraySum = np.concatenate((other_leftUpArray, other_leftDownArray, other_rightUpArray, other_rightDownArray), axis=0)

        # 地块内所有建筑物的顶点（地块内其他建筑物加上当前移动建筑物的顶点）
        leftUpArray = np.insert(other_leftUpArray, a, cur_leftUp, axis=0)
        leftDownArray = np.insert(other_leftDownArray, a, cur_leftDown, axis=0)
        rightUpArray = np.insert(other_rightUpArray, a, cur_rightUp, axis=0)
        rightDownArray = np.insert(other_rightDownArray, a, cur_rightDown, axis=0)

        #由于l此时并不是当前智能体移动的位置，因此，需要计算当前智能体位置
        cur_center_sun_x = (leftDownArray[:, 0] + rightDownArray[:, 0]) / 2
        cur_center_sun = np.concatenate((cur_center_sun_x[:, np.newaxis], leftDownArray[:, 1][:, np.newaxis]), 1)

        # 地块内所有建筑物的日照测试点
        sun_test = np.concatenate((leftDownArray, cur_center_sun, rightDownArray), axis=0)

        # 全部建筑的个数（包括地块内及地块外）
        build_num = len(l_real) + len(no_l)

        # 周边建筑物的顶点坐标
        surrounding_leftUp = []
        surrounding_leftdown = []
        surrounding_rightup = []
        surrounding_rightdown = []
        for i in range(len(l_real), build_num):
            surrounding_leftUp.append(world.landmarks[i].state.p_pos + [-world.landmarks[i].length, world.landmarks[i].width])
            surrounding_leftdown.append(world.landmarks[i].state.p_pos + [-world.landmarks[i].length, 0])
            surrounding_rightup.append(world.landmarks[i].state.p_pos + [world.landmarks[i].length, world.landmarks[i].width])
            surrounding_rightdown.append(world.landmarks[i].state.p_pos + [world.landmarks[i].length, 0])

        # 周边建筑的日照测试点
        surrounding_sun = np.concatenate((surrounding_leftdown, no_l, surrounding_rightdown), axis=0)

        # 所有建筑物的日照测试点
        all_sun_test = np.concatenate((sun_test, surrounding_sun), axis=0)
        # 全部建筑物的四个顶点（地块内所有建筑物的顶点 加上 周边建筑物的顶点）
        all_leftUpArray = np.concatenate((leftUpArray, surrounding_leftUp), axis=0)
        all_leftDownArray = np.concatenate((leftDownArray, surrounding_leftdown), axis=0)
        all_rightUpArray = np.concatenate((rightUpArray, surrounding_rightup), axis=0)
        all_rightDownArray = np.concatenate((rightDownArray, surrounding_rightdown), axis=0)

        # 所有建筑物对应的建筑高度
        new_all_build_high = np.concatenate((high_arrange, high), axis=0)

        # 地块的顶点，必须为封闭区域且逆时针
        land_vertex = [[-0.8, -0.7], [-0.8, 0.7], [0.8, 0.7], [0.8, -0.7],[-0.8, -0.7]][::-1]

        # 绿地顶点（圆形）
        # center = [0,-0.1]  # center point
        # rad = 0.2
        # sep = 2 * math.pi / 60  # sep use 60 angle
        # point = []
        # for angle in range(61):
        #     x = center[0] + rad * math.cos(sep * angle)
        #     y = center[1] + rad * math.sin(sep * angle)
        #     point.append([round(x, 3), round(y, 3)])  # round function  save float with  x.xxx
        # green_vertex = point

        # ================================================退让边界========================================================
        ##当前建筑物的四个顶点有没有出边界
        # land_path = mpltPath.Path(land_vertex)  # matplotlib.path 计算多个点在不在某一多边形内，速度较快
        # cur_inside = land_path.contains_points(cur_l,radius=1e-9)
        #
        # if np.all(cur_inside) == True:  # 矩阵中全部为True，即所有点都在多边形内
        #     bound_res = bound_res - 0
        # else:  # 有一个点不在多边形内
        #     bound_res = bound_res - 10000

        inter_land = 0
        for i in range(len(l)):
            inter_land += Polygon(land_vertex).intersection(Polygon([leftUpArray[i], leftDownArray[i],
                                                                     rightDownArray[i],
                                                                     rightUpArray[i]])).area - Polygon(
                [leftUpArray[i], leftDownArray[i], rightDownArray[i], rightUpArray[i]]).area
        bound_res = bound_res + inter_land * 150 * 150

        # ================================================防火间距========================================================
        rad = 0.09
        sep = 2 * math.pi / 12  # 以30度为切分，分12份
        fire_area = []  # 当前建筑物的防火区域
        for angle in range(4):
            x = cur_rightUp[0] + rad * math.cos(sep * angle)
            y = cur_rightUp[1] + rad * math.sin(sep * angle)
            fire_area.append((float(round(x, 3)), float(round(y, 3))))  # round function  save float with  x.xxx
        for angle in range(3, 7):
            x = cur_leftUp[0] + rad * math.cos(sep * angle)
            y = cur_leftUp[1] + rad * math.sin(sep * angle)
            fire_area.append((float(round(x, 3)), float(round(y, 3))))  # round function  save float with  x.xxx
        for angle in range(6, 10):
            x = cur_leftDown[0] + rad * math.cos(sep * angle)
            y = cur_leftDown[1] + rad * math.sin(sep * angle)
            fire_area.append((float(round(x, 3)), float(round(y, 3))))  # round function  save float with  x.xxx
        for angle in range(9, 13):
            x = cur_rightDown[0] + rad * math.cos(sep * angle)
            y = cur_rightDown[1] + rad * math.sin(sep * angle)
            fire_area.append((float(round(x, 3)), float(round(y, 3))))
        fire_area.append(fire_area[0])  # 多边形需要闭口

        # 新改，重叠面积
        inter_area = 0
        for i in range(len(l) - 1):
            inter_area += Polygon(fire_area).intersection(Polygon(
                [other_leftUpArray[i], other_leftDownArray[i], other_rightDownArray[i], other_rightUpArray[i]])).area
        fire = fire - inter_area * 150 * 150

        # ================================================禁入边界========================================================
        # 当前建筑物的四个顶点不准禁入绿地边界
        # green_path = mpltPath.Path(green_vertex)    # matplotlib.path 计算多个点在不在某一多边形内，速度较快
        # green_inside = green_path.contains_points(cur_l,radius=1e-9)
        # if np.all(green_inside) == True: # 矩阵中全部为True，即所有点都在多边形内
        #     bound_res = bound_res - 200
        # else:                          # 有一个点不在多边形内
        #     bound_res = bound_res - 0

        # inter_green = 0
        # for i in range(len(l)):
        #     inter_green += Polygon(green_vertex).intersection(Polygon(
        #         [leftUpArray[i], leftDownArray[i], rightDownArray[i], rightUpArray[i]])).area
        # bound_green = bound_green - inter_green * 150 * 150

        # ================================================日照约束========================================================

        build_num = len(l_real) + len(no_l)
        all_result = []
        for x in range(build_num):  # 依次选择建筑物
            result = []
            for index in range(len(As)):
                # 所有日照测试点在不在第一栋建筑某一时刻的阴影中
                if index < 4:
                    xx = new_all_build_high[x] / 150 * math.cos(math.pi / 2 - math.radians(As[index])) * Shadow[index]
                    yy = new_all_build_high[x] / 150 * math.sin(math.pi / 2 - math.radians(As[index])) * Shadow[index]
                    stock_building_shadow_leftup = all_leftUpArray[x] + [xx, yy]
                    stock_building_shadow_rightUp = all_rightUpArray[x] + [xx, yy]
                    stock_building_shadow_leftDown = all_leftDownArray[x] + [xx, yy]

                    cur_build_shadow = mpltPath.Path(
                        np.array([all_leftDownArray[x], all_leftUpArray[x], all_rightUpArray[x],
                                  stock_building_shadow_rightUp,
                                  stock_building_shadow_leftup,
                                  stock_building_shadow_leftDown, all_leftDownArray[x]]))
                    cur_inside = cur_build_shadow.contains_points(all_sun_test, radius=1e-9).astype(int)

                    if x < len(l):    ##考虑all_sun_test的顺序问题，将地块内和地块外的循环拆开
                        cur_inside[x] = 0   #当时建筑物的自己的日照测试点参与阴影了，但是不能算被遮挡
                    if x >= len(l):
                        cur_inside[3*len(l)+x-len(l)] = 0

                    result.append(np.array(cur_inside))

                elif index == 4:
                    xx = new_all_build_high[x] / 150 * math.cos(math.pi / 2 - math.radians(As[index])) * Shadow[index]
                    yy = new_all_build_high[x] / 150 * math.sin(math.pi / 2 - math.radians(As[index])) * Shadow[index]
                    stock_building_shadow_leftup = all_leftUpArray[x] + [xx, yy]
                    stock_building_shadow_rightUp = all_rightUpArray[x] + [xx, yy]
                    cur_build_shadow = mpltPath.Path(np.array(
                        [all_leftUpArray[x], all_rightUpArray[x], stock_building_shadow_rightUp,
                         stock_building_shadow_leftup, all_leftUpArray[x]]))
                    cur_inside = cur_build_shadow.contains_points(all_sun_test, radius=1e-9).astype(int)
                    result.append(np.array(cur_inside))

                else:
                    xx = new_all_build_high[x] / 150 * math.cos(math.pi / 2 - math.radians(As[index])) * Shadow[index]
                    yy = new_all_build_high[x] / 150 * math.sin(math.pi / 2 - math.radians(As[index])) * Shadow[index]
                    stock_building_shadow_leftup = all_leftUpArray[x] + [xx, yy]
                    stock_building_shadow_rightUp = all_rightUpArray[x] + [xx, yy]
                    stock_building_shadow_rightDown = all_rightDownArray[x] + [xx, yy]

                    cur_build_shadow = mpltPath.Path(
                        np.array([all_leftUpArray[x], all_rightUpArray[x], all_rightDownArray[x],
                                  stock_building_shadow_rightDown, stock_building_shadow_rightUp,
                                  stock_building_shadow_leftup,
                                  all_leftUpArray[x]]))

                    cur_inside = cur_build_shadow.contains_points(all_sun_test, radius=1e-9).astype(int)  # 也包含计算点在线上的情况
                    if x < len(l):  ##考虑all_sun_test的顺序问题，将地块内和地块外的循环拆开
                        cur_inside[2*len(l)+x] = 0
                    if x >= len(l):
                        cur_inside[3*len(l)+ 2*len(no_l)+ x-len(l)] = 0

                    result.append(np.array(cur_inside))
            all_result.append(result)

        all_shadow_each_time = np.any(all_result, axis=0).astype(int)  # 先横向计算，any防止两个1相加得2  # 9*60
        all_shadow_time = np.sum(all_shadow_each_time, axis=0)  # 竖向计算sum

        # 只优化地块内的所有建筑物的日照测试点时长
        build_black = np.sum(all_shadow_time[:len(l_real) * 3])

        # 计算有几个日照测试点遮挡时长大于7
        # point_num = np.sum((all_shadow_time > 7).astype(int))
        # if point_num == 0:
        #     build_inside = 0
        # else:
        #     build_inside = -40 * point_num

        shadow_time = np.sum(np.where(all_shadow_time > 7, all_shadow_time, 0))
        build_inside = build_inside - shadow_time

        # print(build_black)

        #==================================================reward=======================================================
        reward_p = bound_res + build_inside + fire + bound_green - build_black  # bound_res, fire局部； build_inside，bound_green全局

        reward = bound_res + fire + build_inside + bound_green # 判断是否满足约束

        # 累加奖励值
        reward_sum = fire + bound_res #局部

        if count % fps == 0:
            fream_reward = fream_reward+reward_sum

        if count % fps == 0:
            l[a] = world.agents[0].state.p_pos
            world.landmarks[a].state.p_pos = l[a]
            a =a +1
            if a == len(l):
                a = 0
            world.agents[0].state.p_pos = l_real[a]
        if count%fps_sum== 0 and fream_reward + reward == 0:
            print(l)
            print('shadow_time',build_black)

        if count%fps_sum == 0:
            fream_reward = 0
        return reward_p


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        # 获取此代理的参考框架中所有实体的位置
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos) # - agent.state.p_pos
        return np.concatenate([agent.state.p_vel] + entity_pos)
