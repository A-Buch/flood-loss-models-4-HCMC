# Delete environment
rm(list = ls())


library(corrplot)
library(dplyr)
library(ggplot2)
library(tidyr)

#####Read Data######
#setwd

data1 = readxl::read_xlsx("C:/Users/Anna/Documents/UNI/MA_topic/flood-loss-models-4-HCMC/input_survey_data/all-attributes_shophouses.xlsx")


#### Differentiate between households for which ev1==ev2#######
#Event dates P1Q2.2.1, P1Q2.2.2. If extreme event is also a recent event then both the events are same [ev1=ev2]. Date-mm-dd-yyy
ev1 = data1$P1Q2.2.1
ev2 = data1$P1Q2.2.2
ev1_p = which(as.numeric(data1$P1Q2.2.1)>30000) #to covert from Julian dates
ev2_p = which(as.numeric(data1$P1Q2.2.2)>30000)
for(i in ev1_p){
  dt = as.Date(as.numeric(data1$P1Q2.2.1[i]), origin = "1899-12-30")
  ev1[i] = format(as.Date(dt), "%m/%d/%Y")
}
for(i in ev2_p){
  dt = as.Date(as.numeric(data1$P1Q2.2.2[i]), origin = "1899-12-30")
  ev2[i] = format(as.Date(dt), "%m/%d/%Y")
}
length(which(data1$P1Q2.2.1 == data1$P1Q2.2.2))
plot((as.factor(unlist(data1[,311]))))





#####Damage Variables######
#Loss Influencing variables: 2.3.1, 2.4.1, 2.5.1.x, 2.6.1, 2.7.1.x, 2.8.1.x, 2.9.1, 2.10.1.x + variables for serious events as well
#remove(P1Q2.5.1.88, P1Q2.5.1.99, P1Q2.5.1.specify, P1Q2.7.1.88, P1Q2.7.1.99, P1Q2.7.1.specify, P1Q2.8.1.88, P1Q2.8.1.99, P1Q2.8.1.specify, 
#P1Q2.10.1.88, P1Q2.10.1.99, P1Q2.10.1.specify)

p_var= which(colnames(data1) %in% c('P1Q2.3.1','P1Q2.11.1.99','P1Q2.3.2','P1Q2.11.2.99'))
vars_dam = as.data.frame(c(data1[,p_var[1]:p_var[2]],data1[,p_var[3]:p_var[4]]))
rm_vars = which(colnames(vars_dam) %in% c('P1Q2.5.1.88', 'P1Q2.5.1.99', 'P1Q2.5.1.specify', 'P1Q2.7.1.88', 'P1Q2.7.1.99', 'P1Q2.7.1.specify', 'P1Q2.8.1.88', 'P1Q2.8.1.99', 'P1Q2.8.1.specify', 
                                                'P1Q2.10.1.88', 'P1Q2.10.1.99', 'P1Q2.10.1.specify','P1Q2.11.1.88', 'P1Q2.11.1.99','P1Q2.5.2.88', 'P1Q2.5.2.99', 'P1Q2.5.2.specify', 'P1Q2.7.2.88', 'P1Q2.7.2.99', 'P1Q2.7.2.specify', 'P1Q2.8.2.88', 'P1Q2.8.2.99', 'P1Q2.8.2.specify', 
                                          'P1Q2.10.2.88', 'P1Q2.10.2.99', 'P1Q2.10.2.specify','P1Q2.11.2.88', 'P1Q2.11.2.99'))
vars_dam = vars_dam[,-rm_vars]

## contamnations
vars_dam$P1Q2.5.2.3[which(is.na(vars_dam$P1Q2.5.2.3)==T)] = 0
vars_dam$P1Q2.5.2.4[which(is.na(vars_dam$P1Q2.5.2.4)==T)] = 0
vars_dam$P1Q2.9.1[which(is.na(vars_dam$P1Q2.9.1)==T)] = 99
vars_dam$P1Q2.9.2[which(is.na(vars_dam$P1Q2.9.2)==T)] = 99

## FIXED description part: same matrix indicates 0 when events are different, 1 if events are identical 
## Same matrix indicates 0 for a given damage variable when households have different value for ev1& ev2, 1 when households have same value for ev1&ev2
## It is assumed that if the damage variables value are same for the 2 events then the two events are the same. 
same= matrix(,nrow=252,ncol=40)
for(i in 1:40){
  same[which(as.character(vars_dam[,i]) == as.character(vars_dam[,i+40])),i] = 1
  same[which(as.character(vars_dam[,i]) != as.character(vars_dam[,i+40])),i] = 0
  #same[which(as.numeric(as.character(vars_dam[,i])) != as.numeric(as.character(vars_dam[,i+40]))),i] = 0
}

vars_dam$same = apply(same, 1, function(x) {ifelse(any(x==0), 0, 1)})## sum(pre_vars$same, na.rm= TRUE) = 469 cases of ev1 = ev2 and 531 cases of ev1!=ev2

##In the dataset for loss estimation, count the repeat events as 1 and remove duplicates
#Total events (530*2 + 470*1) -> 1530

#c input dataset: vars_dam upto P1Q2.10; Precautionary measures   
pre_vars = as.data.frame(data1[,c('P2Q1.1.implement', 'P2Q1.2.implement', 'P2Q1.3.implement', 'P2Q1.4.implement', 'P2Q1.5.implement', 'P2Q1.6.implement', 'P2Q1.7.implement')])

##1 - before serious, 2 - before recent, 3 - before both, 4 - after both, 5 - did not implement
#Recent - 1
vars_dam$pre1.1 =0
vars_dam$pre2.1 =0
vars_dam$pre3.1 =0
vars_dam$pre4.1 =0
vars_dam$pre5.1 =0
vars_dam$pre6.1 =0
vars_dam$pre7.1 =0

#Serious - 2
vars_dam$pre1.2 =0
vars_dam$pre2.2 =0
vars_dam$pre3.2 =0
vars_dam$pre4.2 =0
vars_dam$pre5.2 =0
vars_dam$pre6.2 =0
vars_dam$pre7.2 =0

vars_dam$pre1.1[which(pre_vars$P2Q1.1.implement==2)] = 1
vars_dam$pre1.1[which(pre_vars$P2Q1.1.implement==3)] = 1
vars_dam$pre1.2[which(pre_vars$P2Q1.1.implement==1)] = 1
vars_dam$pre1.2[which(pre_vars$P2Q1.1.implement==3)] = 1

vars_dam$pre2.1[which(pre_vars$P2Q1.2.implement==2)] = 1
vars_dam$pre2.1[which(pre_vars$P2Q1.2.implement==3)] = 1
vars_dam$pre2.2[which(pre_vars$P2Q1.2.implement==1)] = 1
vars_dam$pre2.2[which(pre_vars$P2Q1.2.implement==3)] = 1

vars_dam$pre3.1[which(pre_vars$P2Q1.3.implement==2)] = 1
vars_dam$pre3.1[which(pre_vars$P2Q1.3.implement==3)] = 1
vars_dam$pre3.2[which(pre_vars$P2Q1.3.implement==1)] = 1
vars_dam$pre3.2[which(pre_vars$P2Q1.3.implement==3)] = 1

vars_dam$pre4.1[which(pre_vars$P2Q1.4.implement==2)] = 1
vars_dam$pre4.1[which(pre_vars$P2Q1.4.implement==3)] = 1
vars_dam$pre4.2[which(pre_vars$P2Q1.4.implement==1)] = 1
vars_dam$pre4.2[which(pre_vars$P2Q1.4.implement==3)] = 1

vars_dam$pre5.1[which(pre_vars$P2Q1.5.implement==2)] = 1
vars_dam$pre5.1[which(pre_vars$P2Q1.5.implement==3)] = 1
vars_dam$pre5.2[which(pre_vars$P2Q1.5.implement==1)] = 1
vars_dam$pre5.2[which(pre_vars$P2Q1.5.implement==3)] = 1

vars_dam$pre6.1[which(pre_vars$P2Q1.6.implement==2)] = 1
vars_dam$pre6.1[which(pre_vars$P2Q1.6.implement==3)] = 1
vars_dam$pre6.2[which(pre_vars$P2Q1.6.implement==1)] = 1
vars_dam$pre6.2[which(pre_vars$P2Q1.6.implement==3)] = 1

vars_dam$pre7.1[which(pre_vars$P2Q1.7.implement==2)] = 1
vars_dam$pre7.1[which(pre_vars$P2Q1.7.implement==3)] = 1
vars_dam$pre7.2[which(pre_vars$P2Q1.7.implement==1)] = 1
vars_dam$pre7.2[which(pre_vars$P2Q1.7.implement==3)] = 1

##########################################################################################
#Socio-economic variables
vars_soc = data1[,c('P4Q1.1','P4Q1.2','P4Q1.3','P4Q1.4','P4Q1.5.0','P4Q1.5.1','P4Q1.5.2','P4Q1.5.3',
                    'P4Q1.5.4','P4Q1.5.5','P4Q1.5.6','P4Q1.5.7','P4Q1.6','P4Q1.7.1','P4Q1.7.2',
                    'P4Q1.7.3','P4Q1.7.4','P4Q1.7.5','P4Q1.8','P4Q1.9','P4Q1.10')]
vars_soc$P4Q1.6[which(vars_soc$P4Q1.6 == 2)] = 0
vars_soc$P4Q1.9[which(vars_soc$P4Q1.9 == 2)] = 0


#For building variables - during serious and recent events - building age; how long haa the 
#householder lived in the location. LU certificate, building cost.
vars_bui = data1[,c('P4Q2.4','P4Q2.5')]
vars_bui$ba =  as.numeric(gsub(",",".",data1$P4Q2.3))
ev1_year = substr(ev1, nchar(ev1)-4+1, nchar(ev1))
ev2_year = substr(ev2, nchar(ev2)-4+1, nchar(ev2))

#1 - recent
#2 - extreme
data1$P4Q2.1[which(as.numeric(data1$P4Q2.1)==99)] = NA
data1$P4Q2.2[which(as.numeric(data1$P4Q2.2)==99)] = NA

vars_bui$occ_yrs1 = as.numeric(ev1_year)-as.numeric(data1$P4Q2.1)
vars_bui$occ_yrs2 = as.numeric(ev2_year)-as.numeric(data1$P4Q2.1)

vars_bui$bage1 = as.numeric(ev1_year)-as.numeric(data1$P4Q2.2)
vars_bui$bage2 = as.numeric(ev2_year)-as.numeric(data1$P4Q2.2)

#For physical damage, if the householder did not live 
#in this house during the reported flood events, we remove those records from further analysis.
#Valid = 1; not valid = 0
vars_bui$valid1 = 1
vars_bui$valid2 = 1
vars_bui$valid1[which(vars_bui$occ_yrs1 <0)] = 0
vars_bui$valid2[which(vars_bui$occ_yrs2 <0)] = 0

#Renovation: If major renovation -> change bage to that. Otherwise, original bage

ren1 = data1$P4Q4.2.1
ren2 = data1$P4Q4.2.2
ren1[which(ren1 == "99")] = NA
ren2[which(ren2 == "99")] = NA
ren1_p = which(as.numeric(data1$P4Q4.2.1)>40000) #to covert from Julian dates
ren2_p = which(as.numeric(data1$P4Q4.2.2)>40000)

for(i in ren1_p){
  dt = as.Date(as.numeric(data1$P4Q4.2.1[i]), origin = "1899-12-30")
  ren1[i] = format(as.Date(dt), "%m/%d/%Y")
}
for(i in ren2_p){
  dt = as.Date(as.numeric(data1$P4Q4.2.2[i]), origin = "1899-12-30")
  ren2[i] = format(as.Date(dt), "%m/%d/%Y")
}

ren1_year = as.numeric(substr(ren1, nchar(ren1)-4+1, nchar(ren1)))
ren2_year = as.numeric(substr(ren2, nchar(ren2)-4+1, nchar(ren2)))
ren1_year[which(ren1_year == 99)] = NA
ren2_year[which(ren2_year == 99)] = NA
#Recent event
ren1_ev1 = as.numeric(ev1_year)-(ren1_year)
ren2_ev1 = as.numeric(ev1_year)-(ren2_year)
ren1_ev1[which(ren1_ev1 <0)] = NA
ren2_ev1[which(ren2_ev1 <0)] = NA
rec_ren_year = apply(cbind(ren1_ev1,ren2_ev1),1,function(x){return(min(x,na.rm=T))})

#####################################################################################
#####################################################################################

ren1_ev2 = as.numeric(ev2_year)-(ren1_year)
ren2_ev2 = as.numeric(ev2_year)-(ren2_year)
ren1_ev2[which(ren1_ev2 <0)] = NA
ren2_ev2[which(ren2_ev2 <0)] = NA
ext_ren_year = apply(cbind(ren1_ev2,ren2_ev2),1,function(x){return(min(x,na.rm=T))})

ext_ren_year[which(ext_ren_year<0)] = as.numeric(data1$P4Q2.2)[which(ext_ren_year<0)]
ext_ren_year[which(ext_ren_year == 99)] = NA

vars_bui$bage_ren1 =vars_bui$bage1
vars_bui$bage_ren2 =vars_bui$bage2

vars_bui$bage_ren1[which(rec_ren_year !=Inf)] = rec_ren_year[which(rec_ren_year != Inf)]
vars_bui$bage_ren2[which(ext_ren_year != Inf)] = ext_ren_year[which(ext_ren_year != Inf)]
####
vars_bui$bage_ren1[which(vars_bui$bage_ren1 <0)] = NA
vars_bui$bage_ren2[which(vars_bui$bage_ren2 <0)] = NA

vars_bui$bage1[which(vars_bui$bage1 <0)] = NA
vars_bui$bage2[which(vars_bui$bage2 <0)] = NA

##damage variables

# damage_ev1 not used
#damage_ev1 = data1[,c('P1Q2.11.1.1','P1Q2.11.1.2','P1Q2.11.1.3','P1Q2.11.1.4','P1Q2.11.1.5','P1Q2.11.1.6',
#                  'P1Q2.11.1.7','P1Q2.11.1.8','P1Q2.11.1.9','P1Q3.2.1','P1Q3.3.1','P1Q3.4.1','P1Q3.5.1','P1Q3.6.1')]

abs_loss_ev1 = data1$P1Q3.8.1
#Id abs_loss is 2, then no repairs are made. One of the plausible reasons for no repairs is no damage or very minor damage. We replace these 2s by 0. 
#Find zero-loss values
abs_loss_ev1[which((data1$P1Q3.10.1.4==1 | data1$P1Q3.10.1.3==1 | data1$P1Q2.11.1.1 ==1 | (data1$P1Q3.2.1 %in% c(1,99,98) & data1$P1Q3.3.1 %in% c(1,99,98) & 
                                                                                          data1$P1Q3.4.1 %in% c(1,99,98) & data1$P1Q3.5.1 %in% c(1,99,98) & 
                                                                                          data1$P1Q3.6.1 %in% c(1,99,98) & data1$P1Q3.7.1 %in% c(1,99,98))) & data1$P1Q3.8.1 ==2)] = 0

#If no evidence of minor damage or residual damage, approximate it to what would have occured if you repaired the house completely.
abs_loss_ev1[which(abs_loss_ev1==2 & data1$P1Q3.11.1 != 99)] = data1$P1Q3.11.1[which(abs_loss_ev1==2 & data1$P1Q3.11.1 != 99)]*1000000
abs_loss_ev1[which(abs_loss_ev1==99 & data1$P1Q3.11.1 != 99)] = data1$P1Q3.11.1[which(abs_loss_ev1==99 & data1$P1Q3.11.1 != 99)]*1000000
abs_loss_ev1[which(abs_loss_ev1==99)] = NA

#########################################################################################################################################################################################################
#########################################################################################################################################################################################################

# damage_ev2 not used 
#damage_ev2 = data1[,c('P1Q2.11.2.1','P1Q2.11.2.2','P1Q2.11.2.3','P1Q2.11.2.4','P1Q2.11.2.5','P1Q2.11.2.6',
#                      'P1Q2.11.2.7','P1Q2.11.2.8','P1Q2.11.2.9','P1Q3.2.2','P1Q3.3.2','P1Q3.4.2','P1Q3.5.2','P1Q3.6.2')]

abs_loss_ev2 = data1$P1Q3.8.2
#Id abs_loss is 2, then no repairs are made. One of the plausible reasons for no repairs is no damage or very minor damage. We replace these 2s by 0. 
#Find zero-loss values
abs_loss_ev2[which((data1$P1Q3.10.2.4==1 | data1$P1Q3.10.2.3==1 | data1$P1Q2.11.2.1 ==1 | (data1$P1Q3.2.2 %in% c(1,99,98) & data1$P1Q3.3.2 %in% c(1,99,98) & 
                                                                                             data1$P1Q3.4.2 %in% c(1,99,98) & data1$P1Q3.5.2 %in% c(1,99,98) & 
                                                                                             data1$P1Q3.6.2 %in% c(1,99,98) & data1$P1Q3.7.2 %in% c(1,99,98))) & data1$P1Q3.8.2 ==2)] = 0

#If no evidence of minor damage or residual damage, approximate it to what would have occured if you repaired the house completely.
abs_loss_ev2[which(abs_loss_ev2==2 & data1$P1Q3.11.2 != 99)] = data1$P1Q3.11.2[which(abs_loss_ev2==2 & data1$P1Q3.11.2 != 99)]*1000000
abs_loss_ev2[which(abs_loss_ev2==99 & data1$P1Q3.11.2 != 99)] = data1$P1Q3.11.2[which(abs_loss_ev2==99 & data1$P1Q3.11.2 != 99)]*1000000
abs_loss_ev2[which(abs_loss_ev2==99)] = NA

#Building value
#########################
vars_bui$bv = data1$P4Q2.5  # asinged var from 
################
vars_bui$bv[which(vars_bui$bv == 99)] = NA

rloss_ev1 = (abs_loss_ev1/1000000)/vars_bui$bv
rloss_ev1[which(rloss_ev1>1)] = 1

rloss_ev2 = (abs_loss_ev2/1000000)/vars_bui$bv
rloss_ev2[which(rloss_ev2>1)] = 1

###Predictors
colnames(vars_dam) = c('dur1','wd1','con1.0','con1.1','con1.2','con1.3','con1.4','vel1','src1.1','src1.2','src1.3',
                       'war1.1','war1.2','war1.3','war1.4','war1.5','war1.6','war1.7','war1.8','war1.9','war1.10','war_lt1','em1.1','em1.2',
                       'em1.3','em1.4','em1.5','em1.6','em1.7','em1.8','em1.9','dam1.1','dam1.2','dam1.3','dam1.4','dam1.5','dam1.6',
                       'dam1.7','dam1.8','dam1.9',
                       'dur2','wd2','con2.0','con2.1','con2.2','con2.3','con2.4','vel2','src2.1','src2.2','src2.3',
                       'war2.1','war2.2','war2.3','war2.4','war2.5','war2.6','war2.7','war2.7','war2.9','war2.10','war_lt2','em2.1','em2.2',
                       'em2.3','em2.4','em2.5','em2.6','em2.7','em2.8','em2.9','dam2.1','dam2.2','dam2.3','dam2.4','dam2.5','dam2.6',
                       'dam2.7','dam2.8','dam2.9',colnames(vars_dam)[81:95])

vars_bui = vars_bui[,c(1,2,3,6,7,10,11)]
colnames(vars_bui) = c('lu_cert','bv','barea','bage1','bage2','bage_ren1','bage_ren2')

colnames(vars_soc) = c('hh_size','age1','age2','age3','health1','health2','health3','health4','health5',
                       'health6','health7','health8','people_com','org1','org2','org3','org4','org5',
                       'edu','poverty_cert','income')
vars_dam$rloss_1 = rloss_ev1
vars_dam$rloss_2 = rloss_ev2
vars_dam$bloss_1 = abs_loss_ev1
vars_dam$bloss_2 = abs_loss_ev2

vars_dam$id=1:252#1000
# FIXED indices: rloos_1 and bloss_1 to data_ip1, rloss_2 and bloss_2 to ev2
data_ip1 = as.data.frame(vars_dam[,c(1:40,81:88,96,98,100)])
#data_ip1 = as.data.frame(vars_dam[,c(1:40,81:88,96,97,99)])
for( i in 1:nrow(data_ip1)){
  if(data_ip1$same[i]==0){
    sev = as.numeric(vars_dam[i,c(41:80,81,89:95,97,99,100)])
    #sev = as.numeric(vars_dam[i,c(41:80,81,89:96,98,100)])
    data_ip1 = rbind(data_ip1,sev)
  }
}

vars_bui$id = 1:252
data_ip2 = as.data.frame(vars_bui[,c(1:4,6,8)])
for( i in 1:nrow(data_ip2)){
  if(vars_dam$same[i]==0){
    sev = as.numeric(vars_bui[i,c(1:3,5,7,8)])
    data_ip2 = rbind(data_ip2,sev)
  }
}

vars_soc$id = 1:252
data_ip3 = as.data.frame(vars_soc)
for( i in 1:nrow(data_ip3)){
  if(vars_dam$same[i]==0){
    sev = as.numeric(vars_soc[i,])
    data_ip3 = rbind(data_ip3,sev)
  }
}

## Fixed: 
#all_input = cbind(data_ip1[,-which(colnames(data_ip1)=='id')],data_ip2[,-which(colnames(data_ip2)=='id')],data_ip3)
all_input = cbind(
      data_ip1[, colnames(data_ip1)!='id'],
      data_ip2[, colnames(data_ip2)!='id'],
      data_ip3)

all_input$dur1 = as.numeric(as.character(gsub(",",".",all_input$dur1)))
all_input[all_input==99]=NA
all_input$war_lt1 = as.numeric(as.character(gsub(",",".",all_input$war_lt1)))
write.csv(all_input,'C:/Users/Anna/Documents/UNI/MA_topic/flood-loss-models-4-HCMC/input_survey_datar_input_data.csv')


