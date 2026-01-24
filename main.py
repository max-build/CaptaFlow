from datetime import datetime
import os 
import pandas as pd #type:ignore
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ## hide tensorflow messages in interpreter
import asyncio
import websockets #type:ignore
import re
from transformers import pipeline #type:ignore
import re as regex
import time
import logging
import boto3 #type:ignore
from io import StringIO 
import argparse
import credentials

pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest", top_k=None)
##> moved this out of process_batch() to stop python from re-creating the RoBERTa model every single time a new batch is processed. 


##> Schema for pandas dataframe storing insights to be exported
df = pd.DataFrame({
    "timestamp": pd.Series(dtype="datetime64[ns]"),
    "Joy": pd.Series(dtype="float"),
    "Optimism": pd.Series(dtype="float"),
    # "Love": pd.Series(dtype="float"),
    # "Trust": pd.Series(dtype="float"),
    # "Anticipation": pd.Series(dtype="float"),
    "Surprise": pd.Series(dtype="float"),
    "Fear": pd.Series(dtype="float"),
    "Anger": pd.Series(dtype="float"),
    # "Disgust": pd.Series(dtype="float"),
    # "Pessimism": pd.Series(dtype="float"),
    "Sadness": pd.Series(dtype="float"),
    "report_id": pd.Series(dtype="datetime64[ns]")
})


# import boto3 #type:ignore
# from botocore.exceptions import ClientError #type:ignore


# inferences_bank:list = [] ##> list of tuples, stores message inferences that've been lifted from message batches as tuples
##> superior to dataframes as dataframes need to be destroyed and re-created each time something is appended

def main_func(streamer_name):
    with open("files/words_alpha.txt", "r") as words_alpha:
        lines = [k.strip() for k in words_alpha]
        word_db = set(lines) ##> set gives lookups O(1) timing

    def export_to_s3(list_of_tuples):
        if len(list_of_tuples) == 3:
            print("Insights list queue full, load to dataframe.")
            for j in b1.insights_list:
                df.loc[len(df)] = [
                    j[0],
                    round(j[1], 2),
                    round(j[2], 2),
                    round(j[3], 2),
                    round(j[4], 2),
                    round(j[5], 2),
                    round(j[6], 2),
                    j[7] 
                ]
            print(df.head(len(df)))
            bucket = credentials.bucket
            csv_buffer = StringIO()
            df.to_csv(csv_buffer)
            s3_resource = boto3.resource('s3')
            s3_resource.Object(bucket, f'report_{report_id}.csv').put(Body=csv_buffer.getvalue())
            print("Dataframe successfully persisted to S3, clearing b1.insights_list, resetting df.")
            b1.insights_list.clear()
            df[:] = []
        else:
            pass




    ##> report_id stamper, creates column where every row contains this value so AWS quicksuite can differentiate the reports. 
    report_id = datetime.now() ##> fetches timestamp 
    report_id = f"{streamer_name}_{report_id.strftime("%Y-%m-%d_%H:%M:%S")}" # converts timestamp to string

    # async def count_up(): ##> timer, used to detect if chat has slowed down the point the stream can be considered over, triggers dataframe to be pushed to S3. 
    #     for k in range(30, 0, -1):
    #         print(f"Time since last message: {k} seconds. ")
    #         await asyncio.sleep(1)
    #     print("")
    #     # export_data()
    #     # exit()



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

            self._insights_list:list = [] #> list of tuples, stores batch insights in tuple form
            self._message_queue = []

            ##> getters and setters
            ##> getter syntax: print(obj.name)
            ##> setter syntax: obj.name = "John"


        @property
        def insights_list(self):
            return self._insights_list

        @insights_list.setter
        def insights_list(self, insights):
            self._insights_list.append(insights)

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
            # print(f"LOVE score: {round(self.love, 2)}")
            # print(f"TRUST score: {round(self.trust, 2)}")
            print(f"SURPRISE score: {round(self.surprise, 2)}")
            # print(f"ANTICIPATION score: {round(self.anticipation, 2)}")
            print(f"FEAR score: {round(self.fear, 2)}")
            print(f"ANGER score: {round(self.anger, 2)}")
            # print(f"DISGUST score: {round(self.disgust, 2)}")
            # print(f"PESSIMISM score: {round(self.pessimism, 2)}")
            print(f"SADNESS score: {round(self.sadness, 2)}")
            print("\n")
            print((("timestamp"), ("Joy"), ("Optimism"), ("Surprise"), ("Fear"), ("Anger"), ("Sadness"), ("report_id")))


        def reset_message_queue(self):
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
                    if x["label"] == "joy": ## sums cumulative weight scores per each batch so they can be averaged (sentiments are attributes of batch class)
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

            timestamp = datetime.now() # fetches timestamp 
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") # converts timestamp to string

            ##> superior approach (appends batch readings to list of tuples, implement this once pipeline done.)
            # inference_bank.append((timestamp_str, round(b1.joy, 2), round(b1.optimism, 2), round(b1.surprise, 2), round(b1.fear, 2), round(b1.anger, 2), round(b1.sadness, 2), report_id))
        
            
            self.insights_list.append((
                (timestamp_str), 
                round(b1.joy, 2), 
                round(b1.optimism, 2),
                round(b1.surprise, 2), 
                round(b1.fear, 2), 
                round(b1.anger, 2),
                round(b1.sadness, 2),
                (report_id)) 
                )
            ##> check length of insights_list to check if == 30, at that point, add contents to dataframe and persist to S3. 


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


    ##> adds messages to b1's queue 
    def add_to_queue(message): 
        if len(b1.message_queue) < 10:
                b1.message_queue = message # appends message to b1's message queue
                print(f"Current queue size: {len(b1.message_queue)}/10")

        elif len(b1.message_queue) == 10: ## checks length, if 10, processes batch and resets weights/message queue
            print(f"Queue full, processing batch. ")
            b1.process_batch()
            b1.display_results()
            # print(df)
            for t in b1.insights_list:
                print(t)
            print("\nBatch processed. ")
            b1.reset_weights()
            print("Weights reset.")
            b1.reset_message_queue()
            print("Message queue reset.")
            print("\n")

    ##> adds list of tuples to dataframe to be exported to S3
    def add_to_df(list_of_insights):
        pass




    async def main(): 
        ws = await websockets.connect("wss://irc-ws.chat.twitch.tv:443")
        #await ws.send("CAP REQ :twitch.tv/tags twitch.tv/commands")
        await ws.send("NICK justinfan676767")
        await ws.send(f"JOIN #{streamer_name}")
        
        while True:
            msg = await ws.recv()
            if msg.startswith("PING"):
                await ws.send("PONG")
            else:
                try:
                    trim:list = msg.split(":", maxsplit=2) #> returns list
                    if trim[2].startswith("@") or trim[2].startswith("!"):
                        # print(f"Skipped message (command detected): {trim[2]}")
                        pass ##> remove this if behaviour becomes erratic
                    else:
                        # print(f"User message: {trim[2]} ") ##> prints user message (in raw, unnormalised form)
                        trim:list = trim[2].split(" ")
                        dropped_term_counter:int = 0 # counts dropped terms (emote codes, obscure slang etc)
                        for k in range(len(trim)): ##> iterates through individual words in twitch message
                            cleaned_word:str = "".join(regex.findall(r"[a-z]", trim[k].lower())) ##filters out numbers and punctuation from words in twitch message
                            trim[k] = cleaned_word ##> trim[k] cleaned, now only contains letters 
                            if (trim[k]).lower() not in word_db: ##> filters out words in messages which aren't in word_db (dictionary)
                                dropped_term_counter += 1 ##> increments dropped word counter
                                # print(f"dropped terms: {dropped_term_counter}") ##> displays tally of dropped words (emoji codes, slang etc.) from current twitch message
                                trim[k] = "" ##> word in message not in dictionary, its set to blank (means messages with low number of emoji codes can still be processed)

                        if dropped_term_counter >= (len(trim)/2): ##> skips messages where more than half of words are emoji codes or non-words. 
                            # print(f"Message skipped, too many dropped terms.")
                            dropped_term_counter = 0 ##> resets dropped term counter so next message can be processed.
                        # elif len(df) == 2: ##> checks rows of pandas dataframe, exports data as .csv (placeholder, can specify different exit condition later)
                        elif len(b1.insights_list) == 6:
                            export_to_s3(df) ##> persists insights to s3, disabled for now
                            print("Chat ingestion ending, exporting data to S3 bucket ...")
                            exit()
                        else:
                            if trim: ##> checks twitch message (with emoji codes filtered out) is not empty (null)
                                new_message = " ".join(trim) ##> converts trim from list to string (to be processed)
                                add_to_queue(new_message) ##> adds twitch message to queue to be processed in batch.

                except IndexError: ##> catches out of bounds exception thrown, only thrown at start of chat ingestion, could remove this later if needed.
                    print("Index out of bounds caught. ")
                # except Exception:
                #     print("Generic exception caught. ")

    ##> Note: I've commented out print() statements used for indicating when messages were dropped. 

    asyncio.run(main())

    ##> put all code inside of a main(streamer_name) block
    ##> this lets me pass the streamers name into python to specify the streamer without having to manually change the code
    ##> also lets me dynamically write the streamer's name into the report_id column 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("streamer_name", type=str)
    args = parser.parse_args()
    print(f"Commencing sentiment analysis for: {args.streamer_name}")
    main_func(args.streamer_name)


##> next tasks: insights are now stored in tuples and added to dataframe at specified intervals [DONE]
##> i now need to re-think the exit condition as there will be multiple calls to export_to_S3 
    ##> im moving from exporting all of insights in a dataframe at once to doing regular intervals
    ##> of storing them in a dataframe and then exporting those dataframes (maybe 30 batch insights each)














# ## below is template for ingesting twitch chat

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
