library(readxl)
library(dplyr)
library(magrittr)

df01 <- read_xlsx('data/ChengData.xlsx') %>%
  select(CASRN, SMILES, `Experimental Labels`) %>%
  rename(EndPt = `Experimental Labels`) %>%
  na.omit() %>%
  mutate(Source = "Cheng")

write.csv(df01, file = 'data/ChengData_acquire.csv')

df02 <- read_xlsx('data/MansouriData.xlsx') %>%
  select(`CAS-RN`, Smiles, Class) %>%
  rename(CASRN = `CAS-RN`) %>%
  rename(SMILES = Smiles) %>%
  rename(EndPt = Class) %>%
  na.omit() %>%
  mutate(Source = "Mansouri")

write.csv(df02, file = 'data/MansouriData_acquire.csv')