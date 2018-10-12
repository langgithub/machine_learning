"""
@version: 2.0.0
@author: lang
@license: Apache Licence 
@file: nlp03.py
@time: 2018/6/6 15:11

                       .::::.
                     .::::::::.
                    :::::::::::
                ..:::::::::::'
             '::::::::::::'
                .::::::::::
           '::::::::::::::..
                ..::::::::::::.
             ``::::::::::::::::
               ::::``:::::::::'        .:::.              
              ::::'   ':::::'       .::::::::.
            .::::'      ::::     .:::::::'::::.
           .:::'       :::::  .:::::::::' ':::::.
          .::'        :::::.:::::::::'      ':::::.
         .::'         ::::::::::::::'         ``::::.
     ...:::           ::::::::::::'              ``::.
    ```` ':.          ':::::::::'                  ::::..
                       '.:::::'                    ':'````..
"""
from __future__ import unicode_literals # compatible with python3 unicode coding

from deepnlp import nn_parser
import deepnlp
# deepnlp.download()
parser = nn_parser.load_model(name = 'zh')

#Example 1, Input Words and Tags Both
words = ['它', '熟悉', '一个', '民族', '的', '历史']
# words=["富力桃园C区","周边","大型","商业配套","匮乏",'0']
tags = ['r', 'v', 'm', 'n', 'u', 'n']

#Parsing
dep_tree = parser.predict(words, tags)

#Fetch result from Transition Namedtuple
num_token = dep_tree.count()
print ("id\tword\tpos\thead\tlabel")
for i in range(num_token):
    cur_id = int(dep_tree.tree[i+1].id)
    cur_form = str(dep_tree.tree[i+1].form)
    cur_pos = str(dep_tree.tree[i+1].pos)
    cur_head = str(dep_tree.tree[i+1].head)
    cur_label = str(dep_tree.tree[i+1].deprel)
    print ("%d\t%s\t%s\t%s\t%s" % (cur_id, cur_form, cur_pos, cur_head, cur_label))