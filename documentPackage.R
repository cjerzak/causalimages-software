rm(list=ls())
package_name <- "causalimages"
setwd(sprintf("~/Documents/%s-software", package_name))

package_path <- sprintf("~/Documents/%s-software/%s",package_name,package_name)

devtools::document(package_path)
try(file.remove(sprintf("./%s.pdf",package_name)),T)
system(sprintf("R CMD Rd2pdf %s",package_path))

# install.packages( sprintf("~/Documents/%s-software/%s",package_name,package_name),repos = NULL, type = "source")
# library( causalimages ); data(  CausalImagesTutorialData )
log(sort( sapply(ls(),function(l_){object.size(eval(parse(text=l_)))})))

