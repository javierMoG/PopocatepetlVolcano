library(tidyverse)

setwd("~")


base <- read_csv("C:/Users/LENOVO/OneDrive/Documentos/Estancia de Investigación/PopocatepetlVolcano/src/output_outliers.csv")

ggplot(base%>%filter(date<"2023-04-01"), aes(x = date, y = cumsum))+geom_line()+
  geom_vline(xintercept = as.Date("2022-09-24"), 
             color = "red", 
             linetype = "dashed", 
             linewidth = 1)+
  geom_vline(xintercept = as.Date("2022-09-26"), 
             color = "red", 
             linetype = "dashed", 
             linewidth = 1)+
  geom_vline(xintercept = as.Date("2022-11-16"), 
             color = "red", 
             linetype = "dashed", 
             linewidth = 1)



mod_1 <- loess(outliers ~ as.numeric(date), data = base%>%na.omit()%>%filter(date<"2025-09-15"), span = 0.15, degree = 1)

datos_dia <- base%>%na.omit()%>%filter(date<"2025-03-15") |> 
  mutate(ajuste_1 = fitted(mod_1)) |>
  mutate(res_1 = outliers - ajuste_1)

g_1 <- ggplot(datos_dia, aes(x = date)) +
  geom_point(aes(y = outliers), alpha = 0.2, size = 1) +
  geom_line(aes(y = ajuste_1), colour = "red", linewidth = 1.2) + xlab("") +
  labs(caption = "Suavizamiento apropiado")

g_1 


ggplot(base, aes(x = date, y = outliers))+
  geom_line()+geom_point()

acf(base$outliers)
acf(base$outliers, main = "Gráfico de Autocorrelación", lag.max = 815)



serie_anomalias <- ts(base$outliers)

acf(serie_anomalias, lag.max = 815, main = "Autocorrelación de Anomalías Diarias")
pacf(serie_anomalias, lag.max = 815, main = "Autocorrelación Parcial de Anomalías Diarias")

objeto_acf <- acf(serie_anomalias, lag.max = 815, plot = FALSE)

# Calcular el límite de significancia estadística
limite_critico <- 2 / sqrt(length(serie_anomalias))

# Crear tabla con los días y el valor absoluto de su correlación
tabla_valores <- data.frame(
  Dia = reshape::melt(objeto_acf$lag)$value, # Extrae los lags (días)
  Correlacion = objeto_acf$acf
)

# Filtrar para ver cuáles días SÍ tienen correlación importante
dias_con_correlacion <- subset(tabla_valores, abs(Correlacion) > limite_critico)
print(dias_con_correlacion)


# 1. Configuración de parámetros calibrados con tu ACF
ventana_dias <- 84  # 12 semanas de historial
umbral_sd    <- 3  # Límite clásico de 3 desviaciones estándar

# 2. Asegurar formato de vector diario
# (Reemplaza 'tus_datos$anomalias' por tu variable real)
vector_anomalias <- as.numeric(base$outliers)

# 3. Cálculos móviles hacia atrás
media_12sem <- rollapply(vector_anomalias, width = ventana_dias, FUN = mean, fill = NA, align = "right")
sd_12sem    <- rollapply(vector_anomalias, width = ventana_dias, FUN = sd,   fill = NA, align = "right")

# 4. Calcular el límite superior dinámico
limite_superior <- media_12sem + (umbral_sd * sd_12sem)

# 5. Crear el eje de tiempo en días
dias <- 1:length(vector_anomalias)

# 6. Gráfico Principal
# Graficar la serie original en gris para resaltar las alertas
plot(dias, vector_anomalias, type = "l", col = "gray45", lwd = 1,
     main = "Detección de Anomalías con Ventana de 12 Semanas",
     xlab = "Tiempo (Días del Historial)", ylab = "Número de Anomalías",
     panel.first = grid())

# Dibujar la Media Móvil (Comportamiento normal de las últimas 12 semanas)
lines(dias, media_12sem, col = "blue", lwd = 2, lty = 2)

# Dibujar el Umbral Crítico (Línea roja de control)
lines(dias, limite_superior, col = "red", lwd = 2)

# 7. Identificar y pintar con un punto rojo las alertas disparadas
puntos_alerta <- which(vector_anomalias > limite_superior)
points(puntos_alerta, vector_anomalias[puntos_alerta], col = "red", pch = 19, cex = 1.2)

# Leyenda explicativa
legend("topleft", 
       legend = c("Anomalías Diarias", "Media Móvil (84 días)", "Límite Alerta (3 SD)", "Alerta Detectada"),
       col = c("gray45", "blue", "red", "red"), 
       lty = c(1, 2, 1, NA), 
       pch = c(NA, NA, NA, 19), 
       lwd = c(1, 2, 2, NA),
       bty = "n")



# 4. Agregar las nuevas columnas directamente a tu estructura original 'base'
base$Media_Movil      <- media_12sem
base$desviacion_estandar <- sd_12sem
base$Limite_Superior  <- media_12sem + (umbral_sd * sd_12sem)

# 5. Crear la columna lógica de Alerta (TRUE si supera el límite, FALSE si no)
# Usamos un ifelse para que los primeros 84 días (que son NA) queden marcados como FALSE o NA
base$Alerta_Disparada <- ifelse(is.na(base$Limite_Superior), FALSE, vector_anomalias > base$Limite_Superior)


