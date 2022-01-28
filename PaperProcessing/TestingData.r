
library(ggplot2)
library(dplyr)
library(tidyr)

test_data = read.csv("DataTable.csv")

data %>%
  filter(EvalName=="KernelGen")%?%
  select(kernel_mode, map_name, avg_tims)%>%
  

# Baseline
test_data %>%
  filter(EvalName=="BaselineComp")%>%
  select(avg_times, success_rate, test_number, reward, kernel_reward, vehicle, test_number) %>%
  ggplot(aes(x=vehicle, y=(avg_times)))+
  geom_boxplot()+
  geom_point(shape='x', size=6)

# Kernel Reward
test_data %>%
  filter(EvalName=="KernelReward")%>%
  select(avg_times, success_rate, test_number, reward, kernel_reward, sss_reward_scale, test_number) %>%
  unite("label", kernel_reward, sss_reward_scale, sep="__")%>%
  group_by(label) %>%
  mutate(mean_success=mean(success_rate))%>%
  ggplot(aes(x=label, y=avg_times))+
  geom_boxplot(outlier.shape=NA)+
  geom_point(aes(y=avg_times), size=2)+
  geom_point(aes(y=mean_success), shape='x', size=12)

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
  filter(avg_times!=0) %>%
  select(avg_times, success_rate, kernel_reward, vehicle, train_n) %>%
  ggplot(aes(x=train_n, y=avg_times))+
  geom_point()+
  geom_smooth(method=lm)

test_data %>%
  filter(EvalName=="StepsSSS") %>%
  filter(avg_times!=0) %>%
  select(avg_times, success_rate, kernel_reward, vehicle, train_n) %>%
  ggplot(aes(x=train_n, y=success_rate))+
  geom_point()

test_data = read.csv("PaperTable.csv")
# kernel mode
test_data %>%
  filter(EvalName=="LearningMode") %>%
  # filter(avg_times!=0) %>%
  select(avg_times, success_rate, kernel_reward, learning, learning_mode, train_n) %>%
  group_by(learning_mode)%>%
  mutate(mean_time=mean(avg_times))%>%
  ggplot(aes(x=learning_mode, y=avg_times, group=learning_mode))+
  geom_boxplot()+
  geom_point()+
  geom_point(aes(y=mean_time), shape='x', size=10)

