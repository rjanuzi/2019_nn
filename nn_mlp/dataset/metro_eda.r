library(ggplot2)

summary(metro_data)

# Ignorar a coluna holiday (quase nenhum dado)
metro_data$holiday <- factor(metro_data$holiday)
summary(metro_data$holiday)
qplot(x=holiday, data=metro_data)

# Ignorar dados com temperatura < 230 ou > 320
summary(metro_data$temp)
ggplot(metro_data, aes(x=temp)) + 
  geom_histogram(color="black", fill="lightblue", binwidth=1)
ggplot(metro_data, aes(x=temp)) + 
  geom_histogram(color="black", fill="lightblue", binwidth=1) +
  xlim(c(230, 320))

# Ignorar dados com rain_1h > 100
summary(metro_data$rain_1h)
ggplot(metro_data, aes(x=rain_1h)) + 
  geom_histogram(color="black", fill="lightblue", binwidth=0.4) +
  xlim(c(9800, 9850)) +
  ylim(c(0, 100))
ggplot(metro_data, aes(x=rain_1h)) + 
  geom_histogram(color="black", fill="lightblue", binwidth=0.4) +
  xlim(c(0, 100)) +
  ylim(c(0, 100))
  

# Está bem desbalanceado, muitos valores com 0 e pouquissímos maiores que 0
summary(metro_data$snow_1h)
ggplot(metro_data, aes(x=snow_1h)) + 
  geom_histogram(color="black", fill="lightblue", binwidth=0.001)
ggplot(metro_data, aes(x=snow_1h)) + 
  geom_histogram(color="black", fill="lightblue", binwidth=0.001) +
  ylim(c(0,30))

# Ok
summary(metro_data$clouds_all)
ggplot(metro_data, aes(x=clouds_all)) + 
  geom_histogram(color="black", fill="lightblue", binwidth=1)

# Ok (talvez ignorar Smoke e Squall)
metro_data$weather_main <- factor(metro_data$weather_main)
summary(metro_data$weather_main)
qplot(x=weather_main, data=metro_data)

# Ok (Tem alguns com bem poucos valores também)
metro_data$weather_description <- factor(metro_data$weather_description)
summary(metro_data$weather_description)
qplot(x=weather_description, data=metro_data)

# Ok
summary(metro_data$date_time)
summary(metro_data$traffic_volume)
ggplot(metro_data, aes(x=date_time, y=traffic_volume)) +
  geom_line()
