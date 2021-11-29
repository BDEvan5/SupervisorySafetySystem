
library(ggplot2)
library(dplyr)

test_data = read.csv("PaperTable.csv")

test_data %>%
  filter(EvalName=="BaselineComp")%>%
  select(avg_times, success_rate, test_number, reward, kernel_reward, vehicle, test_number) %>%
  ggplot(aes(x=vehicle, y=(avg_times), color=test_number))+
  geom_boxplot()+
  geom_point()

safety <- test_data %>%
  filter(vehicle=="constant"|vehicle=="random")%>%
  filter(test_number==1)%>%
  select(avg_times, success_rate, test_number, vehicle, constant_value, kernel_mode, kernel_filled)

discret <- test_data %>%
  filter(EvalName=="KernelDiscret_ndx")%>%
  # filter(test_number==1)%>%
  select(avg_times, success_rate, test_number, vehicle, constant_value, kernel_mode, kernel_filled)

test_data %>%
  filter(vehicle=="constant"|vehicle=="random")%>%
  filter(test_number==1)%>%
  select(avg_times, success_rate, test_number, vehicle, constant_value, kernel_mode) %>%
  ggplot(aes(x=vehicle, y=avg_times, color=kernel_mode))+
  geom_point()


# make table here


super_reward = test_data %>%
  filter(EvalName=="KernelReward") %>%
  select(avg_times, success_rate, kernel_reward, vehicle, sss_reward_scale) 


test_data %>%
  filter(EvalName=="KernelReward") %>%
  select(avg_times, success_rate, kernel_reward, vehicle, sss_reward_scale) %>%
  ggplot(aes(x=kernel_reward, y=avg_times, size=sss_reward_scale))+
 geom_point()

