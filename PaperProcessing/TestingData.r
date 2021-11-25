test_data = read.csv("DataTable.csv")

library(reshape2)
library(ggplot2)
library(dplyr)

test_data %>%
  filter(reward=="CthRef" | reward=="CthCenter")%>%
  # filter(reward=="CthCenter")%>%
  select(avg_times, success_rate, test_number, r1, reward) %>%
  ggplot(aes(x=r1, y=(avg_times)))+
  geom_point()

test_data %>%
  filter(EvalName=="reward_1"|EvalName=="reward")%>%
  select(avg_times, success_rate, test_number, reward) %>%
  ggplot(aes(x=reward, y=(avg_times)))+
  geom_point()

test_data %>%
  filter(EvalName=="reward_1"||reward=="CthCenter")%>%
  select(avg_times, success_rate, test_number, reward) %>%
  ggplot(aes(x=reward, y=(avg_times)))+
  geom_point()
# ylim(380, 720)


test_data %>%
  filter(EvalName=="PaperTest")%>%
  filter(sss_reward_scale==1)%>%
  filter(kernel_reward=="Magnitude"|kernel_reward=="Constant")%>%
  select(avg_times, success_rate, test_number, reward, kernel_reward, sss_reward_scale, test_number) %>%
  ggplot(aes(x=kernel_reward, y=(avg_times), color=test_number))+
  geom_boxplot()+
  geom_point()

