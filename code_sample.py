# this is part of the code for my summer research
# this is the code to download continous waveform from IRIS and preprocess the data by filtering and downsampling

import os
import urllib
import shutil
import tarfile
import obspy
from obspy.core import read 
from requests_html import HTMLSession

# known a download link, download file to certain folder
def file_download(mseed_url, file_path, day):
    if os.path.exists(file_path + '/*.mseed') == False:

        try:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            os.chdir(file_path)
            file_suffix = os.path.splitext(mseed_url)[1]
            file_name = day + '.tar.mseed'
            urllib.request.urlretrieve(mseed_url, filename = file_name)
            print('download the mseed file successfully')
        except IOError as exception_first:
            print(1, exception_first)

        except Exception as exception_second:
            print(2, exception_second)

    else:
        print('the file has already existed')

# untar the tar.mseed file
def un_tar(mseed_name, file_path):
    tar = tarfile.open(mseed_name)
    names = tar.getnames()
    for name in names:
        tar.extract(name, file_path)
    tar.close()

# move mseed file to file_path
def movefile(srcfile, datfile):
    if not os.path.isfile(srcfile):
        print('%s not exist!') % (srcfile)
    else:
        fpath, fname = os.path.split(datfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.move(srcfile, datfile)

# convert mseed file into sac file
def mseedtosac(datfile):
    st = read(datfile)
    fpath, fname = os.path.split(datfile)   #fpath = file_path
    os.chdir(fpath)
    i = 0
    for tr in st:
        if i != 0 :
            fname = day + '_' + str(i) + '.sac'
        else:
            fname = day + '.sac'

        tr.write(fname, format='SAC')
        i = i + 1

# delete other files, keep sac file and return its name
def cleansac(file_path):
    fpath = file_path
    file_list = os.listdir(fpath)
    for name in file_list:
        file = os.path.join(fpath, name)   # the absolute address of every file
        if not '.sac' in name:
            if os.path.isdir(file):
                shutil.rmtree(file)
            else:
                os.remove(os.path.join(fpath, name))
    
    file_list = os.listdir(fpath)
    filenames = []
    for name in file_list:
        if '.sac' in name:
            file_name = os.path.join(file_path, name)  # the ultimate address of sac file
            filenames.append(file_name)
    return filenames

# filter and downsample
def preprocess(file_name, file_path):
    st = read(file_name)
    tr = st[0]
    tr_new = tr.copy()
    #tr_new.decimate(factor=4)  # downsample

    tr_new.filter('lowpass', freq=20)  # filter
    os.chdir(file_path)
    tr_new.write(file_name, format='SAC')  # save

# get the mseed_url
session = HTMLSession()
url = 'http://ds.iris.edu/pub/userdata/hqZHU/?C=M;O=A'
r = session.get(url)
day = input('input a day (Julia):')
for link in r.html.absolute_links:
    if day in link:
        mseed_url = link
    else:
        continue

file_path = os.path.join('/home/hqzhu/Desktop/T201501/' + day)     # file_path: the address of the ultimate dir

file_download(mseed_url, file_path, day)  # known a download link, download file to certain folder

os.chdir(file_path)
mseed_name = day + '.tar.mseed'
un_tar(mseed_name, file_path)  # untar the tar.mseed file

# move mseed file to file_path
dir_names = os.listdir(file_path)  # obtain names of all directories
for name in dir_names:   # pick up the wanted directory
    if 'tar.mseed' in name:
        continue
    else:
        dirname = name   # dirname is the name of directory which is untared from tar.mseed


srcfile = os.path.join(file_path + '/' + dirname + '/H08S2.IM.mseed')
datfile = os.path.join(file_path + '/H08S2.IM.mseed')  # the ultimate address of mseed file

movefile(srcfile, datfile)

mseedtosac(datfile)  # convert mseed file into sac file

filenames = cleansac(file_path)  # delete other files, keep sac file and return its name

for name in filenames:
    preprocess(name, file_path)  #preprocess
