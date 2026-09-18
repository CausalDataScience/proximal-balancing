## code to prepare `WSCdata` dataset goes here

WSCdata <- read.csv("./data-raw/df2200_cleaned.csv")
usethis::use_data(WSCdata, overwrite = TRUE)

AMAS_WSC <- read.csv("./data-raw/fin_amas.csv")
usethis::use_data(AMAS_WSC, overwrite = TRUE)

BDI_WSC <- read.csv("./data-raw/fin_bdi.csv")
usethis::use_data(BDI_WSC, overwrite = TRUE)

Big5_WSC <- read.csv("./data-raw/fin_big5.csv")
usethis::use_data(Big5_WSC, overwrite = TRUE)

GSES_WSC <- read.csv("./data-raw/fin_gses.csv")
usethis::use_data(GSES_WSC, overwrite = TRUE)

MCS_WSC <- read.csv("./data-raw/fin_mcs.csv")
usethis::use_data(MCS_WSC, overwrite = TRUE)

Math_Pre_WSC <- read.csv("./data-raw/fin_math_pre.csv")
usethis::use_data(Math_Pre_WSC, overwrite = TRUE)

Math_Post_WSC <- read.csv("./data-raw/fin_math_post.csv")
usethis::use_data(Math_Post_WSC, overwrite = TRUE)

Vocab_Pre_WSC <- read.csv("./data-raw/fin_vocab_pre.csv")
usethis::use_data(Vocab_Pre_WSC, overwrite = TRUE)

Vocab_Post_WSC <- read.csv("./data-raw/fin_vocab_post.csv")
usethis::use_data(Vocab_Post_WSC, overwrite = TRUE)

Math_Train_WSC <- read.csv("./data-raw/fin_math_session.csv")
usethis::use_data(Math_Train_WSC, overwrite = TRUE)
