# -*- coding:utf-8 -*-

CLASS_NAMES = [
    "agree"              ,
    "chat_greeting"      ,
    "chat_skip"          ,
    "chat_whoispossessed",
    "chat_whoisseer"     ,
    "chat_whoisvillager" ,
    "chat_whoiswerewolf" ,
    "comingout_possessed",
    "comingout_seer"     ,
    "comingout_villager" ,
    "comingout_werewolf" ,
    "divination"         ,
    "divined_human"      ,
    "divined_werewolf"   ,
    "enum"               ,
    "estimate_possessed" ,
    "estimate_seer"      ,
    "estimate_villager"  ,
    "estimate_werewolf"  ,
    "request_vote"       ,
    "vote"               ,
]
# CLASS_NAMES = [
#     "VOTE",
#     "ESTIMATE",
#     "COMINGOUT",
#     "DIVINED",
#     "DIVINATION"
# ]

CLASS_NAMES_DICT = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))
CLASS_NUM = len(CLASS_NAMES)