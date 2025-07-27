# Notes on Linux

## Disk utility

Often `du` and `df` are not very friendly. 
You want an easy to use utility that will let you know exactly
what is using up space. 

In this case use 
```sh 
ncdu
```

It not available on your system, simply do 
```sh
brew install ncdu
```

### Usage

`ncdu` is very workspace focused so make sure to run it from `~` to show the entire disk usage.
