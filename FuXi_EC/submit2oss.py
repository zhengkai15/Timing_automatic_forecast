# -*-coding:utf-8 -*-
"""
# File       : submit2oss.py
# Time       ：12/29/23 10:59 AM
# Author     ：KaiZheng
# version    ：python 3.8
# Project  : Fuxi-infer-online
# Description：
"""

# -*- coding: utf-8 -*-
import os
import sys
import time
import oss2
import operator
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, wait

_Executor_Pool = ThreadPoolExecutor(max_workers=10)


class Oss_Upload_Download():
    # def __init__(self, bucket_name='fuxi-inference', prefix='inference_output/'):
        # self.auth = oss2.Auth('<AccessKeyId>','<AccessKeySecret>')
        # self.auth = oss2.Auth('LTAI5tCYW8JJzmVZvoWEm9RS', 'NCw1Ez5IblqeOjwrZRmkPZswxSik4y')
        # self.endpoint = 'oss-cn-shanghai.aliyuncs.com'
    def __init__(self, bucket_name='application', prefix='inference_output/'):
        self.auth = oss2.Auth('*', '*')
        self.endpoint = '*.*.*.*'
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name)



    def downloadFiles(self, download_savedir='oss', download_file=[]):
        """ downloadFiles
        download files on the oss
        """
        if not os.path.exists(download_savedir):
            os.makedirs(download_savedir)
            self.download_savedir = download_savedir
            print("The floder {0} is not existing, will be created.".format(
                self.download_savedir))
        download_file = [os.path.join(self.prefix, filename) for filename in download_file]

        def download_file_MPI(tmp_file):
            if not self.bucket.object_exists(tmp_file):
                print("File {0} is not on the OSS!".format(tmp_file))
            else:
                # print("Will download {0} !".format(tmp_file))
                tmp_time = time.time()
                # cut the file name
                filename = tmp_file[tmp_file.rfind("/") + 1:len(tmp_file)]
                localFilename = os.path.join(download_savedir, filename)
                # bucket.get_object_to_file(
                oss2.resumable_download(
                    self.bucket,
                    tmp_file,
                    localFilename,
                    progress_callback=self.percentage)
                # print("\nFile {0} -> {1} downloads finished, cost {2} Sec.".format(
                # tmp_file, localFilename, time.time() - tmp_time ))

        start_time = time.time()
        num = len(download_file)
        future_obj = []
        for i in range(num):
            obj = _Executor_Pool.submit(download_file_MPI, download_file[i])
            future_obj.append(obj)
        wait(future_obj)
        print("All download tasks have finished!")
        print("Cost {0} Sec.".format(time.time() - start_time))

    def uploadFiles(self, upload_file=[],init_step_timestamp = '20231228-06'):
        """
        uploadFiles
        Upload files to the oss
        """
        # start_time = time.time()
        for tmp_file in upload_file:
            if not os.path.exists(tmp_file):
                print("File {0} is not exists!".format(tmp_file))
            else:
                # print("Will upload {0} to the oss!".format(tmp_file))
                # tmp_time = time.time()
                # cut the file name
                filename = tmp_file[tmp_file.rfind("/") + 1:len(tmp_file)]
                ossFilename = os.path.join(self.prefix, init_step_timestamp, filename)
                print(ossFilename)

                oss2.resumable_upload(
                    self.bucket,
                    ossFilename,
                    tmp_file,
                    progress_callback=self.percentage)

                # print("\nFile {0} -> {1} uploads finished, cost {2} Sec.".format(
                #    tmp_file, ossFilename, time.time() - tmp_time ))
        # print("All upload tasks have finished!")
        # print("Cost {0} Sec.".format(time.time() - start_time))

    def showFiles(self):
        """
        Show files on OSS
        """
        print("Show All Files:")
        oss_path_list = []
        for b in islice(
                oss2.ObjectIterator(self.bucket, prefix=self.prefix), None):
            print(b.key)
            oss_path_list.append(b.key)
        return oss_path_list

    def percentage(self, consumed_bytes, total_bytes):
        if total_bytes:
            rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
            print('\r{0}% '.format(rate), end='')
            sys.stdout.flush()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default=None, help='file name to upload')
    parser.add_argument('--prefix', '-prefix', type=str, default='inference_output', help='prefix')
    args = parser.parse_args()

    m = Oss_Upload_Download(prefix=args.prefix)
    try:
        file_names = sorted(os.listdir(args.filename))
        print(file_names)
        init_step_timestamp = args.filename.split('/')[-2][:11]
        file_names = [os.path.join(args.filename, file_name) for file_name in file_names]
        m.uploadFiles(file_names,init_step_timestamp = init_step_timestamp)
    except:
        print(args.filename)
        m.uploadFiles([args.filename])
