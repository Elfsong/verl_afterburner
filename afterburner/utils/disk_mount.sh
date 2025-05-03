echo '[+] install mdadm'
sudo apt install mdadm

echo '[+] unmount disks'
sudo umount /mnt/lucky_space

echo '[+] create RAID array'
sudo mdadm --create /dev/md0 --level=0 --raid-devices=4  /dev/disk/by-id/google-local-nvme-ssd-*

echo '[+] format RAID array'
sudo mkfs.ext4 -F /dev/md0

echo '[+] create disk mount point'
sudo mkdir /mnt/lucky_space

echo '[+] mount disk'
sudo mount /dev/md0 /mnt/lucky_space

echo '[+] grant mount point permission'
sudo chmod a+w /mnt/lucky_space

echo '[+] set HF environments'
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/mnt/lucky_space/

echo '[+] Done'