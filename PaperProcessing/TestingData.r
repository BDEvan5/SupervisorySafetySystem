
library(ggplot2)
library(dplyr)
library(tidyr)

test_data = read.csv("PaperTable.csv")

# Baseline
test_data %>%
  filter(EvalName=="BaselineComp")%>%
  select(avg_times, success_rate, test_number, reward, kernel_reward, vehicle, test_number) %>%
  ggplot(aes(x=vehicle, y=(avg_times)))+
  geom_boxplot()+
  geom_point(shape='x', size=6)

# Kernel Reward
test_data %>%
  filter(EvalName=="KernelReward"&test_number==1)%>%
  # filter(sss_reward_scale==0.2|sss_reward_scale==1)%>%
  filter(sss_reward_scale==0.2)%>%
  select(avg_times, success_rate, test_number, reward, kernel_reward, sss_reward_scale, test_number) %>%
  group_by(kernel_reward) %>%
  mutate(mean_time=mean(avg_times))%>%
  ggplot(aes(x=kernel_reward))+
  geom_boxplot(aes(y=avg_times))+
  geom_point(aes(y=avg_times), size=2)+
  geom_point(aes(y=mean_time), shape='x', size=10)

test_data %>%
  filter(EvalName=="KernelReward"&test_number==1)%>%
  filter((kernel_reward=="Magnitude"&sss_reward_scale==1)|(kernel_reward=="Constant"&sss_reward_scale==0.5))%>%
  select(avg_times, success_rate, test_number, reward, kernel_reward, vehicle, test_number) %>%
  ggplot(aes(x=kernel_reward, y=(avg_times)))+
  geom_boxplot()+
  geom_point(shape='x', size=6)

test_data %>%
  filter(EvalName=="KernelReward"&test_number==1&kernel_reward=="Constant")%>%
  select(avg_times, success_rate, test_number, reward, kernel_reward, sss_reward_scale, test_number) %>%
  group_by(sss_reward_scale)%>%
  mutate(mean_time=mean(avg_times))%>%
  ggplot(aes(x=sss_reward_scale))+
  geom_boxplot(aes(y=avg_times, group=sss_reward_scale))+
  geom_point(aes(y=avg_times), size=2)+
  geom_line(aes(y=mean_time))+
  geom_point(aes(y=mean_time), shape='x', size=8)

test_data %>%
  filter(EvalName=="KernelReward"&test_number==1&kernel_reward=="Magnitude")%>%
  select(avg_times, success_rate, test_number, reward, kernel_reward, sss_reward_scale, test_number) %>%
  group_by(sss_reward_scale)%>%
  mutate(mean_time=mean(avg_times))%>%
  mutate(median_time=median(avg_times))%>%
  ggplot(aes(x=sss_reward_scale))+
  geom_boxplot(aes(y=avg_times, group=sss_reward_scale))+
  geom_point(aes(y=avg_times), size=2)+
  geom_line(aes(y=mean_time))+
  geom_point(aes(y=mean_time), shape='x', size=8)


test_data %>%
  filter(EvalName=="KernelReward"&test_number==1)%>%
  select(avg_times, success_rate, test_number, reward, kernel_reward, sss_reward_scale, test_number) %>%
  ggplot(aes(x=sss_reward_scale, y=(avg_times), group=kernel_reward))+
  # geom_line(aes(linetype=kernel_reward))+
  geom_point(aes(color=kernel_reward), shape='x', size=6)


test_data %>%
  filter(EvalName=="KernelReward") %>%
  select(avg_times, success_rate, kernel_reward, vehicle, sss_reward_scale) %>%
  ggplot(aes(x=kernel_reward, y=avg_times, size=sss_reward_scale))+
  geom_point()


# Discretisation
test_data %>%
  filter(EvalName=="KernelDiscret_ndx")%>%
  filter(success_rate>80)%>%
  select(avg_times, success_rate, test_number, n_dx, constant_value, kernel_mode, kernel_filled)%>%
  ggplot(aes(x=n_dx, y=kernel_filled, group=kernel_mode))+
  geom_line(aes(linetype=kernel_mode))+
  geom_point()
test_data %>%
  filter(EvalName=="KernelDiscret_ndx")%>%
  select(avg_times, success_rate, test_number, n_dx, constant_value, kernel_mode, kernel_filled)%>%
  ggplot(aes(x=n_dx, y=success_rate, group=kernel_mode))+
  geom_line(aes(linetype=kernel_mode))

test_data %>%
  filter(vehicle=="constant"|vehicle=="random")%>%
  filter(test_number==1)%>%
  select(avg_times, success_rate, test_number, vehicle, constant_value, kernel_mode) %>%
  ggplot(aes(x=vehicle, y=avg_times, color=kernel_mode))+
  geom_point()




# Steps SSS
test_data %>%
  filter(EvalName=="StepsSSS") %>%
  filter(test_number==3|test_number==4) %>%
  filter(avg_times!=0) %>%
  select(avg_times, success_rate, kernel_reward, vehicle, train_n) %>%
  ggplot(aes(x=train_n, y=avg_times))+
  geom_point()+
  geom_smooth(method=lm)

test_data %>%
  filter(EvalName=="StepsSSS") %>%
  filter(test_number==3|test_number==4) %>%
  filter(avg_times!=0) %>%
  select(avg_times, success_rate, kernel_reward, vehicle, train_n) %>%
  ggplot(aes(x=train_n, y=success_rate))+
  geom_point()

test_data = read.csv("PaperTable.csv")
# kernel mode
test_data %>%
  filter(EvalName=="LearningMode") %>%
  filter(test_number==6) %>%
  # filter(avg_times!=0) %>%
  select(avg_times, success_rate, kernel_reward, learning, learning_mode, train_n) %>%
  mutate(mean_time=mean(avg_times))%>%
  ggplot(aes(x=learning_mode, y=avg_times, group=learning_mode))+
  geom_boxplot()+
  geom_point()
  # geom_point(aes(y=mean_time), shape='x', size=10)

learning = test_data %>%
  filter(EvalName=="LearningMode") %>%
  # filter(avg_times!=0) %>%
  select(avg_times, success_rate, kernel_reward, learning, test_number, name)
