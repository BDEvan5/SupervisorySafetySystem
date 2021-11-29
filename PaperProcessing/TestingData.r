test_data = read.csv("PaperTable.csv")

library(ggplot2)
library(dplyr)


test_data %>%
  filter(EvalName=="BaselineComp")%>%
  select(avg_times, success_rate, test_number, reward, kernel_reward, vehicle, test_number) %>%
  ggplot(aes(x=vehicle, y=(avg_times), color=test_number))+
  geom_boxplot()+
  geom_point()

