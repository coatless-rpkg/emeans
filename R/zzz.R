.onLoad <- function(libname, pkgname) {
  try_register <- function(...) {
    tryCatch(register_tidyclust_engine(), error = function(e) NULL)
    invisible(NULL)
  }
  if (isNamespaceLoaded("tidyclust")) {
    try_register()
  }
  setHook(packageEvent("tidyclust", "onLoad"), try_register)
}
