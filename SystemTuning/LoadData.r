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