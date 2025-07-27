# Mac 

## VM on MAC


We can run Linux on Mac using OrbStack. 
However the issue with OrbStack is that it does not allow us to run
a GUI. 
Another MAC app that allows us to use the GUI is [UTM](https://github.com/utmapp/UTM).


### Install UTM


```sh 
brew install --cask utm
```

This installs `UTM` and will be accessible on Mac using the Command Palette as well.
So do `Cmd + Space` and launch UTM.

### Using UTM

There are lots of distributions already available. 
Open UTM and then click on `Browse` which opens their webpage. 
Click on your desired version and then click on `Open in UTM`.


### Arch on UTM

If I install Arch on UTM using the default Arch UTM there is an `alarm login` prompt. 
Use `alarm` as username and password.

Next, the install finishes but there's nothing on Arch. So we first start by installing basic utilities

```sh 
# Install wget
su - # Password is root

pacman -Syu # Update arch
pacman -S wget
```
