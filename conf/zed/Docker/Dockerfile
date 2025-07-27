# Use a standard Ubuntu image as the base
FROM ubuntu:latest

# Install OpenSSH server and sudo
RUN apt-get update && apt-get install -y openssh-server sudo

# Create a non-root user for SSH access
RUN useradd -m -s /bin/bash zed
RUN echo "zed:zed" | chpasswd
RUN adduser zed sudo

# Set up SSH for the zed user
RUN mkdir -p /home/zed/.ssh
COPY authorized_keys /home/zed/.ssh/authorized_keys
RUN chown -R zed:zed /home/zed/.ssh
RUN chmod 700 /home/zed/.ssh
RUN chmod 600 /home/zed/.ssh/authorized_keys

# Create sshd privilege separation directory
RUN mkdir -p /run/sshd

# Expose the SSH port
EXPOSE 22

# Start the SSH service
CMD ["/usr/sbin/sshd", "-D"]
