from datetime import datetime
import os 
import pandas as pd #type:ignore
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ## hide tensorflow messages in interpreter
import asyncio
import websockets #type:ignore
import re
from transformers import pipeline
from nltk.corpus import words #type:ignore


# TODO:
    # take messages from twitch into list_test (process_batch function)
    # remove emotes from strings (using nltk.words), maybe 'for word in phrase, if word not in nltk.words ...' 
    # enable iterative functionality (so twitch messages are added to dataframe ongoingly)
    # export dataframe to external source
    # find way to note per batch what was happening in the stream (what the streamer said etc.)

class batch():
    def __init__(self):
        ##> store accumulative weights from sentiments for when averages are calculated
        self._joy = 0.0
        self._optimism = 0.0
        self._love = 0.0
        self._trust = 0.0
        self._anticipation = 0.0
        self._surprise = 0.0
        self._fear = 0.0
        self._anger = 0.0
        self._disgust = 0.0
        self._pessimism = 0.0
        self._sadness = 0.0

        self._message_queue = ["I can’t stop smiling after that news.",
                                "I feel uneasy about how things turned out",
                                "That moment filled me with pure frustration.",
                                "I’m so proud of how far I’ve come today.",
                                "This whole situation makes me deeply sad.",
                                "I’m buzzing with excitement right now.",
                                "I feel oddly nostalgic thinking about that.",
                                "I’m confused and not sure what to believe.",]

        ##> getters and setters
        ##> getter syntax: print(obj.name)
        ##> setter syntax: obj.name = "John"

    @property
    def message_queue(self):
        return self._message_queue
    
    @message_queue.setter
    def message_queue(self, message):
        self._message_queue.append(message)

    @property ##> property is getter
    def joy(self):
        return self._joy

    @joy.setter ##> var.setter is setter
    def joy(self, joy):
        self._joy = joy

    @property
    def optimism(self):
        return self._optimism
    
    @optimism.setter
    def optimism(self, optimism):
        self._optimism = optimism

    @property
    def love(self):
        return self._love

    @love.setter
    def love(self, love):
        self._love = love

    @property
    def trust(self):
        return self._trust

    @trust.setter
    def trust(self, trust):
        self._trust = trust

    @property
    def anticipation(self):
        return self._anticipation

    @anticipation.setter
    def anticipation(self, anticipation):
        self._anticipation = anticipation

    @property 
    def surprise(self):
        return self._surprise

    @surprise.setter
    def surprise(self, surprise):
        self._surprise = surprise

    @property
    def fear(self):
        return self._fear

    @fear.setter
    def fear(self, fear):
        self._fear = fear

    @property
    def anger(self):
        return self._anger

    @anger.setter
    def anger(self, anger):
        self._anger = anger

    @property
    def disgust(self):
        return self._disgust

    @disgust.setter
    def disgust(self, disgust):
        self._disgust = disgust

    @property
    def pessimism(self):
        return self._pessimism

    @pessimism.setter
    def pessimism(self, pessimism):
        self._pessimism = pessimism

    @property
    def sadness(self):
        return self._sadness

    @sadness.setter
    def sadness(self, sadness):
        self._sadness = sadness


    def display_results(self):
        print("\n------- BATCH RESULTS -------")
        print(f"JOY score: {round(self.joy, 2)}")
        print(f"OPTIMISM score: {round(self.optimism, 2)}")
        print(f"LOVE score: {round(self.love, 2)}")
        print(f"TRUST score: {round(self.trust, 2)}")
        print(f"SURPRISE score: {round(self.surprise, 2)}")
        print(f"ANTICIPATION score: {round(self.anticipation, 2)}")
        print(f"FEAR score: {round(self.fear, 2)}")
        print(f"ANGER score: {round(self.anger, 2)}")
        print(f"DISGUST score: {round(self.disgust, 2)}")
        print(f"PESSIMISM score: {round(self.pessimism, 2)}")
        print(f"SADNESS score: {round(self.sadness, 2)}")

    def clear_message_queue(self):
        self.message_queue.clear()

    def process_batch(self): ## this needs to ingest twitch chat and add messages to list_test so they can be processed
        counter = 0
        pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest", top_k=None) 
        if self.message_queue:
            sentiments = pipe(self.message_queue) ## pipe object passes arguments to model, sentiments is [[{}]] -> list[list[dictionary{}]]
        for k in sentiments: # iterates through inner list
            print("\n")
            print(f"Message: {self.message_queue[counter]}") ## prints messages (they occur in same order in list and in sentiments, this puts messages next to the sentiment printout)
            counter += 1 ##counter increments per each loop so I can print sentiment and score at that index position. 
            for x in k: ## within inner list, iterates through dictionaries (containing sentiment scores)
                if x["label"] == "joy":
                    self.joy += x["score"]
                elif x["label"] == "optimism":
                    self.optimism += x["score"]
                elif x["label"] == "love":
                    self.love += x["score"]
                elif x["label"] == "trust":
                    self.trust += x["score"]
                elif x["label"] == "anticipation":
                    self.anticipation += x["score"]
                elif x["label"] == "surprise":
                    self.surprise += x["score"]
                elif x["label"] == "fear":
                    self.fear += x["score"]
                elif x["label"] == "anger":
                    self.anger += x["score"]
                elif x["label"] == "disgust":
                    self.disgust += x["score"]
                elif x["label"] == "pessimism":
                    self.pessimism += x["score"]
                elif x["label"] == "sadness":
                    self.sadness += x["score"]
                print(x)

        ##> once weights summed, averages them
        self.joy = self.joy/len(self.message_queue)
        self.optimism = self.optimism/len(self.message_queue)
        self.love = self.love/len(self.message_queue)
        self.trust = self.trust/len(self.message_queue)
        self.anticipation = self.anticipation/len(self.message_queue)
        self.surprise = self.surprise/len(self.message_queue)
        self.fear = self.fear/len(self.message_queue)
        self.anger = self.anger/len(self.message_queue)
        self.disgust = self.disgust/len(self.message_queue)
        self.pessimism = self.pessimism/len(self.message_queue)
        self.sadness = self.sadness/len(self.message_queue)

    def reset_weights(self):
        self.joy = 0.0
        self.optimism = 0.0
        self.love = 0.0
        self.trust = 0.0
        self.anticipation = 0.0
        self.surprise = 0.0
        self.fear = 0.0
        self.anger = 0.0
        self.disgust = 0.0
        self.pessimism = 0.0
        self.sadness = 0.0

b1 = batch()

b1.process_batch() # core processing logic
b1.display_results() # display results to see output

##> Schema for pandas dataframe storing insights to be exported
df = pd.DataFrame({
    "timestamp": pd.Series(dtype="datetime64[ns]"),
    "Joy": pd.Series(dtype="float"),
    "Optimism": pd.Series(dtype="float"),
    "Love": pd.Series(dtype="float"),
    "Trust": pd.Series(dtype="float"),
    "Anticipation": pd.Series(dtype="float"),
    "Surprise": pd.Series(dtype="float"),
    "Fear": pd.Series(dtype="float"),
    "Anger": pd.Series(dtype="float"),
    "Disgust": pd.Series(dtype="float"),
    "Pessimism": pd.Series(dtype="float"),
    "Sadness": pd.Series(dtype="float"),
})

timestamp = datetime.now() # fetches timestamp 
timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") # converts timestamp to string
# print(f"Timestamp: [{timestamp_str}]") # prints timestamp




## adds new row to dataframe
## len(df) used to specify the row number im updating
## len(df) will always point to the next row (which is always empty)
df.loc[len(df)] = [
    timestamp_str, 
    round(b1.joy, 2), 
    round(b1.optimism, 2), 
    round(b1.love, 2), 
    round(b1.trust, 2), 
    round(b1.anticipation, 2), 
    round(b1.surprise, 2), 
    round(b1.fear, 2), 
    round(b1.anger, 2), 
    round(b1.disgust, 2), 
    round(b1.pessimism, 2), 
    round(b1.sadness, 2)
    ]

b1.reset_weights()

## adds new row to dataframe
## len(df) used to specify the row number im updating
## len(df) will always point to the next row (which is always empty)


print(df)


# ##> needs to check the length of the batch message_queue to assess if there are enough messages to trigger sentiment analysis
# if len(b1.message_queue) == 5:
#     process_batch(b1.message_queue, joy, optimism, love, trust, anticipation, surprise, fear, anger, disgust, pessimism, sadness)
# else:
#     print(f"Messages in queue: {len(b1.message_queue)}")







# df = pd.DataFrame({
#     "timestamp": pd.Series(dtype="datetime64[ns]"),
#     "Joy": pd.Series(dtype="float"),
#     "Optimism": pd.Series(dtype="float"),
#     "Love": pd.Series(dtype="float"),
#     "Trust": pd.Series(dtype="float"),
#     "Anticipation": pd.Series(dtype="float"),
#     "Surprise": pd.Series(dtype="float"),
#     "Fear": pd.Series(dtype="float"),
#     "Anger": pd.Series(dtype="float"),
#     "Disgust": pd.Series(dtype="float"),
#     "Pessimism": pd.Series(dtype="float"),
#     "Sadness": pd.Series(dtype="float"),
# })



# BELOW IS CODE FOR INGESTING TWITCH CHAT

# message_queue:list = [] ## current limit 10


# def add_to_queue(message): ## (in progress)
#     if len(list_test) < 5:
#             list_test.append(message)
#             print(f"Current queue size: {len(message_queue)}/5")

#     elif len(message_queue) == 5:
#         print(f"Queue full, emptying queue")
#         process_batch(list_test)
#         list_test.clear()
#     ## if len(message_queue) == 10:
#         ## message_queue.analyse_sentiment()
#         ## message_queue.wipe()


# async def main(): 
#     ws = await websockets.connect("wss://irc-ws.chat.twitch.tv:443")
#     #await ws.send("CAP REQ :twitch.tv/tags twitch.tv/commands")
#     await ws.send("NICK justinfan39485")
#     await ws.send("JOIN #asmongold247")
    
#     while True:
#         msg = await ws.recv()
#         if msg.startswith("PING"):
#             await ws.send("PONG")
#         else:
#             try:
#                 trim:list = msg.split(":", maxsplit=2) #> returns list
#                 if trim[2].startswith("@") or trim[2].startswith("!"):
#                     print(f"Skipped message (command detected): {trim[2]}")
#                 else:
#                     print(f"User message: {trim[2]} ") ## print how many messages are in message queue (ie. 0/10, 1/10 etc ongoingly)


#             except IndexError:
#                 print("Index out of bounds caught. ")


# asyncio.run(main())

## get timestamp
## use pandas to create dataframe
## use timestamp as primary key in dataframe
    # > maybe each batch can be 
## 1 row = 1 batch (columns are each sentiment) (timestamp as primary key)
    # > maybe each column can be most dom


## TESTING PRINTING FUNCTIONS

# print("\n------- RESULTS -------")
# print(f"\nBatch joy score: {joy/len(list_test)} | Total joy score: {joy} | Joy score rounded: {round(joy/len(list_test), 2)}")
# print(f"\nBatch optimism score: {optimism/len(list_test)} | Total optimism score: {optimism} | optimism score rounded: {round(optimism/len(list_test), 2)}")
# print(f"\nBatch love score: {love/len(list_test)} | Total love score: {love} | love score rounded: {round(love/len(list_test), 2)}")
# print(f"\nBatch trust score: {trust/len(list_test)} | Total trust score: {trust} | trust score rounded: {round(trust/len(list_test), 2)}")
# print(f"\nBatch anticipation score: {anticipation/len(list_test)} | Total anticipation score: {anticipation} | anticipation score rounded: {round(anticipation/len(list_test), 2)}")
# print(f"\nBatch surprise score: {surprise/len(list_test)} | Total surprise score: {surprise} | surprise score rounded: {round(surprise/len(list_test), 2)}")
# print(f"\nBatch fear score: {fear/len(list_test)} | Total fear score: {fear} | fear score rounded: {round(fear/len(list_test), 2)}")
# print(f"\nBatch anger score: {anger/len(list_test)} | Total anger score: {anger} | anger score rounded: {round(anger/len(list_test), 2)}")
# print(f"\nBatch disgust score: {disgust/len(list_test)} | Total disgust score: {disgust} | disgust score rounded: {round(disgust/len(list_test), 2)}")
# print(f"\nBatch pessimism score: {pessimism/len(list_test)} | Total pessimism score: {pessimism} | pessimism score rounded: {round(pessimism/len(list_test), 2)}")
# print(f"\nBatch sadness score: {sadness/len(list_test)} | Total sadness score: {sadness} | sadness score rounded: {round(sadness/len(list_test), 2)}")

# once this is complete, work out how to timestamp batches. 













## ORIGINAL FUNCTION

# import asyncio
# import websockets
# import re

# phrase_queue:list = []

# async def main():
#     ws = await websockets.connect("wss://irc-ws.chat.twitch.tv:443")
#     #await ws.send("CAP REQ :twitch.tv/tags twitch.tv/commands")
#     await ws.send("NICK justinfan39485")
#     await ws.send("JOIN #nezba")
    
#     while True:
#         msg = await ws.recv()
#         if msg.startswith("PING"):
#             await ws.send("PONG")
#         else:
#             try:
#                 trim = msg.split(":", maxsplit=2)
#                 print(f"User message: {trim[2]}")
#             except IndexError:
#                 print("Index out of bounds caught. ")
#                 #m = re.search(r"@?[^\s:]*\s*:(\w+)![^:]+:(.+)", msg)
#                 #if m:
#                 #    print(f"{m.group(1)}: {m.group(2)}")

# asyncio.run(main())


# r_joy = round(joy/len(list_test), 2)
# r_optimism = round(optimism/len(list_test), 2)
# r_love = round(love/len(list_test), 2)
# r_trust = round(trust/len(list_test), 2)
# r_anticipation = round(anticipation/len(list_test), 2)
# r_surprise = round(surprise/len(list_test), 2)
# r_fear = round(fear/len(list_test), 2)
# r_anger = round(anger/len(list_test), 2)
# r_disgust = round(disgust/len(list_test), 2)
# r_pessimism = round(pessimism/len(list_test), 2)
# r_sadness = round(sadness/len(list_test), 2)






### old (non-iterative) working solution to perform analysis on list_test (no twitch ingestion)


# from datetime import datetime
# import os 
# import pandas as pd #type:ignore
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ## hide tensorflow messages in interpreter
# import asyncio
# import websockets #type:ignore
# import re
# from transformers import pipeline
# from nltk.corpus import words #type:ignore




# # TODO:
#     # take messages from twitch into list_test (process_batch function)
#     # enable iterative functionality (so twitch messages are added to dataframe ongoingly)
#     # export dataframe to external source
#     # find way to note per batch what was happening in the stream (what the streamer said etc.)
  

# def process_batch(message_list:list): ## this needs to ingest twitch chat and add messages to list_test so they can be processed
#     pass



# list_test:list = [
#     "I feel so proud of my work today. TwitchConHYPE TwitchConHYPE pepeHands monkaS catchJAM",
#     "This news has left me uneasy now.",
#     "I’m thrilled things went so well.",
#     "I can’t stop laughing at this.",
#     "My heart feels heavy tonight.",
#     "I’m calm and content right now.",
#     "Everything about this scares me.",
#     "I’m frustrated beyond belief.",
#     "I feel hopeful about tomorrow.",
#     "That moment filled me with awe.",
#     "I’m disappointed with myself.",
#     "This made me oddly sentimental.",
#     "I’m anxious about the results.",
#     "I feel deeply touched by this.",
#     "I’m confused by what happened."
# ]


# # Sentiment table, stores cumulative values of sentiments as weights are tallied. 

# joy:float = 0
# optimism:float = 0
# love:float = 0
# trust:float = 0
# anticipation:float = 0
# surprise:float = 0
# fear:float = 0
# anger:float = 0
# disgust:float = 0
# pessimism:float = 0
# sadness:float = 0

# score_list = [joy, optimism, love, trust, anticipation, surprise, fear, anger, disgust, pessimism, sadness]

# ## timestamp needed for plotting sentiments on chart

# ## this block prints all messages from list_test as well as their ranked sentiments identified in that message
# counter = 0
# pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest", top_k=None) 
# if list_test:
#     sentiments = pipe(list_test) ## pipe object passes arguments to model, sentiments is [[{}]] -> list[list[dictionary{}]]
# for k in sentiments: # iterates through inner list
#     print("\n")
#     print(f"Message: {list_test[counter]}") ## prints messages (they occur in same order in list and in sentiments, this puts messages next to the sentiment printout)
#     counter += 1 ##counter increments per each loop so I can print sentiment and score at that index position. 
#     for x in k: ## within inner list, iterates through dictionaries (containing sentiment scores)
#         if x["label"] == "joy":
#             joy += x["score"]
#         elif x["label"] == "optimism":
#             optimism += x["score"]
#         elif x["label"] == "love":
#             love += x["score"]
#         elif x["label"] == "trust":
#             trust += x["score"]
#         elif x["label"] == "anticipation":
#             anticipation += x["score"]
#         elif x["label"] == "surprise":
#             surprise += x["score"]
#         elif x["label"] == "fear":
#             fear += x["score"]
#         elif x["label"] == "anger":
#             anger += x["score"]
#         elif x["label"] == "disgust":
#             disgust += x["score"]
#         elif x["label"] == "pessimism":
#             pessimism += x["score"]
#         elif x["label"] == "sadness":
#             sadness += x["score"]
#         print(x)

# ## fetch the sentiment label with the heaviest weight for that batch. 


# ## aquires average of each sentiment weighting
# ## then rounds them to 2 decimals for comprehensible viewing 
# r_joy = round(joy/len(list_test), 2)
# r_optimism = round(optimism/len(list_test), 2)
# r_love = round(love/len(list_test), 2)
# r_trust = round(trust/len(list_test), 2)
# r_anticipation = round(anticipation/len(list_test), 2)
# r_surprise = round(surprise/len(list_test), 2)
# r_fear = round(fear/len(list_test), 2)
# r_anger = round(anger/len(list_test), 2)
# r_disgust = round(disgust/len(list_test), 2)
# r_pessimism = round(pessimism/len(list_test), 2)
# r_sadness = round(sadness/len(list_test), 2)



# ## Displays sentiment scores per batch 
# print("\n------- BATCH RESULTS -------")
# print(f"JOY score: {r_joy}")
# print(f"OPTIMISM score: {r_optimism}")
# print(f"LOVE score: {r_love}")
# print(f"TRUST score: {r_trust}")
# print(f"ANTICIPATION score: {r_anticipation}")
# print(f"SURPRISE score: {r_surprise}")
# print(f"FEAR score: {r_fear}")
# print(f"ANGER score: {r_anger}")
# print(f"DISGUST score: {r_disgust}")
# print(f"PESSIMISM score: {r_pessimism}")
# print(f"SADNESS score: {r_sadness}")
# # print(f"DOMINANT SENTIMENT: {dominant_sentiment}")
# timestamp = datetime.now()
# timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
# print(f"Timestamp: [{timestamp_str}]")


# ## get timestamp
# ## use pandas to create dataframe
# ## use timestamp as primary key in dataframe
#     # > maybe each batch can be 
# ## 1 row = 1 batch (columns are each sentiment) (timestamp as primary key)
#     # > maybe each column can be most dom


# ## TESTING PRINTING FUNCTIONS

# # print("\n------- RESULTS -------")
# # print(f"\nBatch joy score: {joy/len(list_test)} | Total joy score: {joy} | Joy score rounded: {round(joy/len(list_test), 2)}")
# # print(f"\nBatch optimism score: {optimism/len(list_test)} | Total optimism score: {optimism} | optimism score rounded: {round(optimism/len(list_test), 2)}")
# # print(f"\nBatch love score: {love/len(list_test)} | Total love score: {love} | love score rounded: {round(love/len(list_test), 2)}")
# # print(f"\nBatch trust score: {trust/len(list_test)} | Total trust score: {trust} | trust score rounded: {round(trust/len(list_test), 2)}")
# # print(f"\nBatch anticipation score: {anticipation/len(list_test)} | Total anticipation score: {anticipation} | anticipation score rounded: {round(anticipation/len(list_test), 2)}")
# # print(f"\nBatch surprise score: {surprise/len(list_test)} | Total surprise score: {surprise} | surprise score rounded: {round(surprise/len(list_test), 2)}")
# # print(f"\nBatch fear score: {fear/len(list_test)} | Total fear score: {fear} | fear score rounded: {round(fear/len(list_test), 2)}")
# # print(f"\nBatch anger score: {anger/len(list_test)} | Total anger score: {anger} | anger score rounded: {round(anger/len(list_test), 2)}")
# # print(f"\nBatch disgust score: {disgust/len(list_test)} | Total disgust score: {disgust} | disgust score rounded: {round(disgust/len(list_test), 2)}")
# # print(f"\nBatch pessimism score: {pessimism/len(list_test)} | Total pessimism score: {pessimism} | pessimism score rounded: {round(pessimism/len(list_test), 2)}")
# # print(f"\nBatch sadness score: {sadness/len(list_test)} | Total sadness score: {sadness} | sadness score rounded: {round(sadness/len(list_test), 2)}")

# # once this is complete, work out how to timestamp batches. 







# ## BELOW IS CODE FOR INGESTING TWITCH CHAT

# # message_queue:list = [] ## current limit 10

# # def process_sentiment(list_of_messages): ## function for passing queued messages for sentiment analysis

# # def add_to_queue(message): ## (in progress)
# #     if len(message_queue) < 10:
# #             message_queue:list.append(message)
# #             print(f"Current queue size: {len(message_queue)}/10")

# #     elif len(message_queue) == 10:
# #         print(f"Queue full, emptying queue")
# #         message_queue
# #     ## if len(message_queue) == 10:
# #         ## message_queue.analyse_sentiment()
# #         ## message_queue.wipe()


# # async def main(): 
# #     ws = await websockets.connect("wss://irc-ws.chat.twitch.tv:443")
# #     #await ws.send("CAP REQ :twitch.tv/tags twitch.tv/commands")
# #     await ws.send("NICK justinfan39485")
# #     await ws.send("JOIN #asmongold247")
    
# #     while True:
# #         msg = await ws.recv()
# #         if msg.startswith("PING"):
# #             await ws.send("PONG")
# #         else:
# #             try:
# #                 trim:list = msg.split(":", maxsplit=2) #> returns list
# #                 if trim[2].startswith("@") or trim[2].startswith("!"):
# #                     print(f"Skipped message (command detected): {trim[2]}")
# #                 else:
# #                     print(f"User message: {trim[2]} ") ## print how many messages are in message queue (ie. 0/10, 1/10 etc ongoingly)


# #             except IndexError:
# #                 print("Index out of bounds caught. ")


# # asyncio.run(main())









# ## ORIGINAL FUNCTION

# # import asyncio
# # import websockets
# # import re

# # phrase_queue:list = []

# # async def main():
# #     ws = await websockets.connect("wss://irc-ws.chat.twitch.tv:443")
# #     #await ws.send("CAP REQ :twitch.tv/tags twitch.tv/commands")
# #     await ws.send("NICK justinfan39485")
# #     await ws.send("JOIN #nezba")
    
# #     while True:
# #         msg = await ws.recv()
# #         if msg.startswith("PING"):
# #             await ws.send("PONG")
# #         else:
# #             try:
# #                 trim = msg.split(":", maxsplit=2)
# #                 print(f"User message: {trim[2]}")
# #             except IndexError:
# #                 print("Index out of bounds caught. ")
# #                 #m = re.search(r"@?[^\s:]*\s*:(\w+)![^:]+:(.+)", msg)
# #                 #if m:
# #                 #    print(f"{m.group(1)}: {m.group(2)}")

# # asyncio.run(main())


# # r_joy = round(joy/len(list_test), 2)
# # r_optimism = round(optimism/len(list_test), 2)
# # r_love = round(love/len(list_test), 2)
# # r_trust = round(trust/len(list_test), 2)
# # r_anticipation = round(anticipation/len(list_test), 2)
# # r_surprise = round(surprise/len(list_test), 2)
# # r_fear = round(fear/len(list_test), 2)
# # r_anger = round(anger/len(list_test), 2)
# # r_disgust = round(disgust/len(list_test), 2)
# # r_pessimism = round(pessimism/len(list_test), 2)
# # r_sadness = round(sadness/len(list_test), 2)


# df = pd.DataFrame({
#     "timestamp": pd.Series(dtype="datetime64[ns]"),
#     "Joy": pd.Series(dtype="float"),
#     "Optimism": pd.Series(dtype="float"),
#     "Love": pd.Series(dtype="float"),
#     "Trust": pd.Series(dtype="float"),
#     "Anticipation": pd.Series(dtype="float"),
#     "Surprise": pd.Series(dtype="float"),
#     "Fear": pd.Series(dtype="float"),
#     "Anger": pd.Series(dtype="float"),
#     "Disgust": pd.Series(dtype="float"),
#     "Pessimism": pd.Series(dtype="float"),
#     "Sadness": pd.Series(dtype="float"),
# })

# ## adds new row to dataframe
# ## len(df) used to specify the row number im updating
# ## len(df) will always point to the next row (which is always empty)
# df.loc[len(df)] = [timestamp_str, r_joy, r_optimism, r_love, r_trust, r_anticipation, r_surprise, r_fear, r_anger, r_disgust, r_pessimism, r_sadness]


# print(df)