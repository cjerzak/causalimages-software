package_name <- "causalimages"
setwd(sprintf("~/Documents/%s-software", package_name))

package_path <- sprintf("~/Documents/%s-software/%s",package_name,package_name)

devtools::document(package_path)
try(file.remove(sprintf("./%s.pdf",package_name)),T)
system(paste(shQuote(file.path(R.home("bin"), "R")), "CMD", "Rd2pdf", shQuote(package_path)))

#install.packages(package_path)
#install.packages( sprintf("~/Documents/%s-software/%s",package_name,package_name),repos = NULL, type = "source")


