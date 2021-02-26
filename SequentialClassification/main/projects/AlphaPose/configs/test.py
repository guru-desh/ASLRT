import os, glob
import string

abs_path = '/path/to/Alphapose/dataset'
users = os.listdir(os.path.abspath(abs_path))

for user in users:
    user = os.path.join(abs_path, user)
    if os.path.isdir(user):
        result = [y for x in os.walk(user) for y in glob.glob(os.path.join(x[0], '*.json'))]
        for res in result:

            '''
            name = os.path.split(res)[1]
            head = os.path.split(res)[0]
            if 'alphapose' not in name:
                name = 'alphapose_' + name
                new = os.path.join(head, name)
                os.rename(res, new)
            
            windows_name = res.split('\\')[-1]
            if '-20' not in windows_name and '-21' not in windows_name:
                name = os.path.split(res)[1]
                head = os.path.split(res)[0]
                if 'Alphapose' in os.path.join(*head.split('\\')[-3:]):
                    trial = '_'.join(head.split('\\')[-2:])
                else:
                    trial = '_'.join(head.split('\\')[-3:])
                name = 'alphapose_' + trial + name[9:]
                new = os.path.join(head, name)
                print(res, new)
                #os.rename(res, new)
            '''
            if res.count('-21')>0:
                fname = os.path.split(res)[1]
                idx = fname.index('Depth')
                prev = fname[:idx]
                fname = fname[idx + 5:]
                if 'p5' not in fname:
                    while fname[0] not in string.ascii_letters:
                        fname = fname[1:]
                else:
                    fname = fname[7:]
                    while fname[0] not in string.ascii_letters:
                        fname = fname[1:]
                fname = os.path.join(os.path.split(res)[0], prev + '.' + fname)
                print(fname)
                #os.rename(res, fname)
                    

            
