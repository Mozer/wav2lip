from flask import (
    Flask,
    jsonify,
    request,
    Response,
    render_template_string,
    abort,
    send_from_directory,
    send_file,
)
from flask_cors import CORS, cross_origin

import modules.wav2lip.wav2lip_module as wav2lip_module
import json, os, random, glob, time, re
from pathlib import Path
from types import SimpleNamespace
import cv2
import time

import pyaudio
import wave
import os
import tempfile

# globals
pyaudio_p = None
next_video_chunk_global = 1
player_loop_running = 0.0
os.environ["face_detect_running"] = "0"

wav2lip_is_busy = 0
wav2lip_has_fresh_video = 0
wav2lip_next_chunk_to_gen = 0
parent_dir = os.path.dirname(os.path.abspath(__file__))

# Used for making text xml compatible, needed for voice pitch and speed control
table = str.maketrans({
    "<": "&lt;",
    ">": "&gt;",
    "&": "&amp;",
    "'": "&apos;",
    '"': "&quot;",
})

def xmlesc(txt):
    return txt.translate(table)
    

print("Initializing wav2lip module")



# generate and save video, returns nothing
# @audio: filename, wav file without extension
# @order: rand, latest. order to choose a file in a folder. use 'latest' to animate latest SD image generation, point SD output path to some folder in /modules/wav2lip/input/[char_folder]
# @chunk: chunk must be > 0. if chunk==0 filename will be wav2lip.mp4 (old way); otherwise: wav2lip_[chunk].mp4
# @reply_part: 0 - it is the first audio for current text reply, for syncing
def wav2lip_server_generate(char_folder="default", device="cpu", audio="test", order="rand", chunk=0, reply_part=0):

    global wav2lip_is_busy, wav2lip_has_fresh_video, wav2lip_next_chunk_to_gen, next_video_chunk_global, player_loop_running
    steps_tried = 0
    chunk = int(chunk)
    reply_part = int(reply_part)
    
    if (audio!="silence" and time.time() - player_loop_running > 3.0):
        print("wav2lip player loop is not found ("+str(player_loop_running)+"), starting it")
        wav2lip_server_play_init()    
    
    # new reply came, ignore prev
    if (reply_part == 0):
        wav2lip_next_chunk_to_gen = chunk
        next_video_chunk_global = chunk
        #print("part_0 reply came, setting globals wav2lip_next_chunk_to_gen and to: "+str(chunk))
    
    print("in wav2lip_server_generate: is busy: "+str(wav2lip_is_busy)+", face_detect_running: "+str(os.environ["face_detect_running"])+", chunk: "+str(chunk)+", chunk_needed: "+str(wav2lip_next_chunk_to_gen)+", reply: "+str(reply_part))
    
    folder_path = "modules/wav2lip/input/"+char_folder+"/"
    if not os.path.isdir(folder_path):
        print("video folder "+char_folder+"("+folder_path+") is not found, switching to 'default'")
        char_folder = "default"        
    
    # waiting loop, all chunks must be in asc order
    while steps_tried < 300: # 30 s
        if (not wav2lip_is_busy and not int(os.environ["face_detect_running"])):
            if (chunk == wav2lip_next_chunk_to_gen or not wav2lip_next_chunk_to_gen): # needed chunk or first run
                break   # break is to break from waiting, not for skipping
                
        #print("wav2lip_is_busy: "+str(wav2lip_is_busy)+", chunk: "+str(chunk)+", chunk_needed: "+str(wav2lip_next_chunk_to_gen)+",  sleeping for 1 s")
        time.sleep(0.1)
        steps_tried += 1
        
    # work    
    if (not wav2lip_is_busy and not int(os.environ["face_detect_running"])): 
        wav2lip_is_busy = 1
        
        xtts_play_allowed = xtts_play_allowed_check()
        if (xtts_play_allowed):        
            
            wav2lip_has_fresh_video = 0
            
            # wav2lip_batch_size was 32, 64 eats more vram but faster
            wav2lip_args_json = '''{
                "checkpoint_path": "modules/wav2lip/checkpoints/wav2lip.pth", 
                "face": "modules/wav2lip/input/default/", 
                "audio":"test.wav", 
                "outfile":"modules/wav2lip/output/wav2lip.mp4", 
                "img_size":96, 
                "fps":15, 
                "wav2lip_batch_size":32,
                "box":[-1, -1, -1, -1], 
                "face_det_batch_size":4,
                "pads":[0, 10, 0, 0],
                "crop":[0, -1, 0, -1], 
                "nosmooth": "False", 
                "resize_factor":1, 
                "rotate":"False",
                "device":"cpu"}'''
            wav2lip_args = json.loads(wav2lip_args_json, object_hook=lambda d: SimpleNamespace(**d))

            folder_path = "modules/wav2lip/input/"+char_folder+"/"
            files = [ f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path,f)) and f != 'silence.mp4' and not f.endswith(".json") ]
            if len(files) < 1:
                print("no files in input/"+char_folder+". Put some videos there.")
                return "True"
            if order=="latest":
                filename = max(files, key=lambda x: os.stat(folder_path+x).st_mtime)
                filename = Path(filename).name
            else:
                rand_r = random.randrange(0, len(files))
                filename = files[rand_r]
            
            wav2lip_args.face = folder_path+filename
            if (chunk):
                wav2lip_args.outfile = "modules/wav2lip/output/wav2lip_"+str(chunk)+".m3u8"  # not used anymore
            else:
                wav2lip_args.outfile = "modules/wav2lip/output/wav2lip.mp4"
            #print("wav2lip is using char_folder "+char_folder+" and "+order+" order, starting with file: "+filename+". Audio file: "+audio+".wav. Output: "+ wav2lip_args.outfile)    
            wav2lip_args.device = device
            wav2lip_args.chunk = chunk
            if (audio == "silence"):
                wav2lip_args.audio = audio+".wav" # silence init request
            else:            
                if os.path.isfile(os.path.join("music.wav")):
                    wav2lip_args.audio = "music.wav" # testing music and wav2ip
                    print("found music.wav. using it instead of TTS audio")           
                elif os.path.isfile(os.path.join("music.mp3")):
                    wav2lip_args.audio = "music.mp3" # testing music and wav2ip
                    print("found music.mp3. using it instead of TTS audio")
                else:
                    wav2lip_args.audio = "tts_out/"+audio+".wav" # test.wav from silero and out.wav from xttsv2        
            
            wav2lip_module.wav2lip_main(wav2lip_args)
            wav2lip_has_fresh_video = 1
            if (not reply_part):
                next_video_chunk_global = chunk # first part of reply
                #print("changed next_video_chunk_global: "+str(next_video_chunk_global))
            wav2lip_next_chunk_to_gen = chunk + 1
            #print("done with wav2lip, next_chunk_needed: "+str(wav2lip_next_chunk_to_gen))       
        
        else:
            print("speech detected, wav2lip_server won't generate")       
        
        wav2lip_is_busy = 0         
        
    else:        
        print("Error: skipping some mp4 and setting busy to 0 after timeout. wav2lip was busy: "+str(wav2lip_is_busy)+", chunk: "+str(chunk)+", chunk_needed: "+str(wav2lip_next_chunk_to_gen)+", step: "+str(steps_tried))
        wav2lip_is_busy = 0
   
    return "True"


# return created video as mp4 using http, not used now
def wav2lip_server_play(fname, char_folder):
    if fname == "silence":
        WAV2LIP_OUTPUT_PATH = os.path.join(parent_dir, "input", char_folder, "")
    else:
        WAV2LIP_OUTPUT_PATH = os.path.join(parent_dir, "output", "")    
    print(WAV2LIP_OUTPUT_PATH)
    print(fname)
    return send_from_directory(WAV2LIP_OUTPUT_PATH, f"{fname}.mp4")
    
# SET and load silero language model
def wav2lip_server_silero_set_lang(tts_service, fname):
    tts_service.load_model(lang_model=fname+".pt")
    print("silero model "+fname+".pt loaded ")
    return "True"

# GET char folders in input folder
def wav2lip_server_get_chars():
    folders = [ f for f in os.listdir("modules/wav2lip/input/") if os.path.isdir(os.path.join("modules/wav2lip/input/",f)) ]
    return folders

# OLD. don't use
# silero tts generate (override)
def wav2lip_server_tts_generate(tts_service, voice):
    print("in wav2lip_server_tts_generate")
    print(voice)
    if "text" not in voice or not isinstance(voice["text"], str):
        abort(400, '"text" is required')
    if "speaker" not in voice or not isinstance(voice["speaker"], str):
        abort(400, '"speaker" is required')
    if "voice_speed" not in voice or not isinstance(voice["voice_speed"], str):
        voice['voice_speed'] = 'medium'
    if "voice_pitch" not in voice or not isinstance(voice["voice_pitch"], str):
        voice['voice_pitch'] = 'medium'
        
    # Remove asterisks
    voice['text'] = voice['text'].replace("*", "")
    try:
        # Remove the destination file if it already exists
        #if os.path.exists('../../test.wav'):
        #    os.remove('../../test.wav')
        prosody = '<prosody rate="{}" pitch="{}">'.format(voice['voice_speed'], voice['voice_pitch'])
        silero_input = '<speak>'+prosody+xmlesc(voice['text'])+'</prosody></speak>'
        audio = tts_service.generate(voice['speaker'], silero_input)
        #audio_file_path = os.path.join(os.path.dirname(os.path.abspath(audio)), os.path.basename(audio))
        audio_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"+os.path.basename(audio))
        #audio_file_path = +audio_file_path
        print("audio: "+str(audio))
        print("audio_file_path: "+str(audio_file_path))
        #os.rename(audio, audio_file_path)
        return send_file(audio_file_path, mimetype="audio/x-wav")
    except Exception as e:
        print(e)
        abort(500, voice['speaker'])
        
    return "True"

# deletes all files (not folders) in folder
def delete_all_old_files(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                print("del "+file_path)
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# find new files in folder, excluding _tmp files
def check_for_new_file(folder):
    list_of_files = [filename for filename in os.listdir(folder) if '_tmp' not in filename]
    sorted_paths = []
    for file in list_of_files:
        sorted_paths.append(os.path.abspath(os.path.join(folder, file)))
    if (len(sorted_paths)):
        filename = max(sorted_paths, key=lambda x: os.stat(x).st_mtime)
        filename = Path(filename).name    
        if len(filename):
            return filename
        else:
            return ""
    return ""  

def get_shared_temp_value():
    file_path = os.path.join(tempfile.gettempdir(), "xtts_play_allowed.txt")
    try:
        with open(file_path, "r") as file:
            value = file.read().strip()
        return value
    except IOError:
        print(f"Error reading from file: {file_path}")
        return None
    
# NEW, temp/xtts_play_allowed.txt for '0' or '1'
# @filename: not used, we store file in /temp
# returns: int 0 - stop, 1 - play allowed
def xtts_play_allowed_check():
    #Check if var contains '0', stop streaming.
    value = get_shared_temp_value()
    if (value is None): # var not set
        value = 1        
    else:
        value = int(value)        
    
    return value   

# get all char folder with videos, sorted by date desc
def wav2lip_server_get_chars_desc():
    base_dir = r'modules\wav2lip\input'

    # Get a list of all items in the base directory
    dirs = wav2lip_server_get_chars()

    # sort by modification time
    #dirs = [d for d in items if isdir(join(base_dir, d))]
    dirs.sort(key=lambda x: os.path.getmtime(os.path.join(base_dir, x)), reverse=True)
    
    return dirs
    

# download checkpoint if needed, make face-detect for newly added video if found
def wav2lip_server_generate_dummy():
    char_folders = wav2lip_server_get_chars_desc()
    print("wav2lip: running init generation with "+char_folders[0]+" and silence.wav")
    wav2lip_server_generate(char_folders[0], "cuda", "silence", "latest", 0, 0)
    
    

# wav2lip player loop
# plays latest found videos locally, using cv2 and pyAudio
# wav2lip_has_fresh_video is a global flag: 0/1, set in wav2lip_server_generate()
def wav2lip_server_play_init():
    
    global wav2lip_has_fresh_video, next_video_chunk_global, player_loop_running
    
    player_loop_running = time.time()
    
    print("Deleting old temporary wavs and mp4s.")    
    video_path = 'modules/wav2lip/temp/' # Path where you want to check for existence of videos
    audio_path = 'tts_out/' # Path where you want to check for existence of audio
    delete_all_old_files(audio_path)
    delete_all_old_files(video_path)
    next_video_chunk_global = 1  
    step = 0
    xtts_play_allowed = 1
    
    captions = ["Video chat", "Skynet", "San-Ti", "Waifu", "Interdimensional TV", "Mozer TV"]
    rand_caption = random.choice(captions)
    
    cv2.destroyAllWindows()    # prev windows if any
    print("\nWav2lip videos can be played now.\n\n")    
    
    while True:

        if (wav2lip_has_fresh_video or not step % 50): # if flag or every 2s
            
            # known filename
            video_file = f"result_{str(next_video_chunk_global)}.mp4"
            video_file_path = os.path.join(video_path, video_file)
            audio_file = f"out_{str(next_video_chunk_global)}.wav"
            audio_file_path = os.path.join(audio_path, audio_file)    
            
            if os.path.exists(video_file_path):
                if os.path.exists(audio_file_path):
                    print(str(time.time())+" calling play_video_with_audio: "+video_file_path+", next_video_chunk_global: "+str(next_video_chunk_global))
                    wav2lip_has_fresh_video = 0
                    
                    xtts_play_allowed = xtts_play_allowed_check()
                    if (xtts_play_allowed): # call player                         
                        next_video_chunk_global = play_video_with_audio(video_file_path, audio_file_path, True, next_video_chunk_global, rand_caption)
                        print("done with play_video_with_audio, latest next_video_chunk_global to play: "+str(next_video_chunk_global))  
                    else:
                        print("speech detected, wav2lip won't play "+str(next_video_chunk_global)+".mp4")
                        next_video_chunk_global += 1
                else:
                    print("Error: video file "+video_file+" exists, but audio "+audio_file+" doesn't")
                    time.sleep(1)
        
        # just check all new vids in dir    
        if (not step % 50): # every 2s
            latest_file = check_for_new_file(video_path)
            #print("checking all new files in dir, latest_file_found: "+latest_file)
            if (len(latest_file)):
                next_chunk_found = int(re.findall('\d+', str(latest_file))[0])
                if (next_chunk_found > next_video_chunk_global):
                    next_video_chunk_global = next_chunk_found
                    print("next_chunk found: " + str(next_video_chunk_global))
                    next_video_chunk_global = int(next_video_chunk_global)
                    next_video_chunk_global += 1            
                    video_file = f"result_{str(next_video_chunk_global)}.mp4"
                    video_file_path = os.path.join(video_path, video_file)
                    audio_file = f"out_{str(next_video_chunk_global)}.wav"
                    audio_file_path = os.path.join(audio_path, audio_file)
            #print("sleep 1, "+video_file_path+" is not found")
            
        if (xtts_play_allowed):
            time.sleep(0.04)    # normal sleep
        else:
            time.sleep(0.01)    # speech detected - lets clean all pending threads - dont sleep too much
        
        step+=1
        player_loop_running = time.time()
        
    player_loop_running = 0.0 


# PLAYER
# play given video and further video if it exixts
# @video_file: full path
# @audio_file: full path
# returns next chunk number to play, thay doesn't now (int)
def play_video_with_audio(video_file, audio_file, use_pyaudio=True, start_chunk=0, caption='Video chat'):
    
    global pyaudio_p, player_loop_running
    
    # async fix between video and audio, in bytes. 2400 bytes = 0.10s, can be negative. 
    #sync_audio_delta_bytes = 5000 # 5000 for bluetooth
    sync_audio_delta_bytes = 0 # 0 for wired speakers
    current_chunk = int(start_chunk) # video number
    current_audio_chunk = 0 # chunk in current audio
    videos_played_n = 0
    last_auidio_time = 0.0
    audio_path = "tts_out/"
    
    print(str(time.time())+" in play_video_with_audio, video: "+video_file + ", current_video_chunk: "+str(current_chunk))
   
    # Create a resizable window
    cv2.namedWindow(caption, cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if (not fps): 
        fps = 25
    time_between_video_frames = int(1000 / fps) # 40 ms for 25fps
    audio_is_behind = 1 #  audio is always lagging behind wav2lip video    
    #print(str(time.time())+" after cv2 cap")
        
    while os.path.isfile(video_file) and os.path.isfile(audio_file):        
        
        #print(str(time.time())+" before pyaudio")
        # Open a PyAudio stream
        if (pyaudio_p is None):
            pyaudio_p = pyaudio.PyAudio() # takes 0.10 s        
        wf = wave.open(audio_file, 'rb')
        stream = pyaudio_p.open(format=pyaudio_p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        #print(str(time.time())+" after open")
        buffer_size = int(round((wf.getframerate() / 1000) * time_between_video_frames)) # 960 bytes == 40 ms between frames in a 25fps video        
        #print(str(time.time())+" after buffer_size")
        # Read the first audio file into the buffer
        data = wf.readframes(buffer_size)
        if (audio_is_behind):
            #print(str(time.time())+" starting playback of "+str(current_chunk-1))
            stream.write(data)
            data = wf.readframes(int(buffer_size/2 + sync_audio_delta_bytes))
        current_chunk+=1
        start_time = time.time()
        step = 0
        
        while True:
            
            if (step % 5 == 0): # check once in 0.15s
                xtts_play_allowed = xtts_play_allowed_check()
                if (xtts_play_allowed == 0):
                    print("speech detected, play_video_with_audio stopped")
                    return current_chunk # won't play, exiting
            step += 1
            
            # Play audio buffer 
            stream.write(data) #blocking. takes ~0.03s            
            
            # play 1 video frame
            if cap.isOpened():            
                ret, frame = cap.read()
                if not ret: # last video frame
                    print("cv2: missing video frame")
                if (ret):
                    cv2.imshow(caption, frame) # window caption, image
                elapsed = (time.time() - start_time) * 1000  # msec
                play_time = int(cap.get(cv2.CAP_PROP_POS_MSEC)) + time_between_video_frames # video 25 fps: 40 ms
                #sleep = max(1, int(play_time - elapsed)) + sync_delta_ms
                #if (sleep < 1):
                #    sleep = 1
                #print(str(time.time())+" sleep "+str(sleep)+", start_time:"+str(start_time)+", v_play_t: "+str(play_time)+", elapsed: "+str(elapsed))
                sleep = 1
                if cv2.waitKey(sleep) & 0xFF == ord("q"):
                    break
            else:
                print("Notice: cv2 is not opened for "+video_file+" (doesn't exist yet)")
                if (os.path.isfile(video_file)):
                    cap.release()
                    cap = cv2.VideoCapture(video_file)
                    start_time = time.time()
                    #print ("new "+video_file + " is opened after Frame Drop")

            # Check if there are more audio frames to play. 1920 bytes is normally returned for buffer_size 960 for 25fps video
            if not len(data) and current_audio_chunk > 0:                
                
                # Check if the next audio file exists
                audio_file = audio_path+"out_"+str(current_chunk)+".wav"
                if os.path.exists(audio_file):                    
                    wf.close() # Close the current audio file                     
                    wf = wave.open(audio_file, 'rb')  # Open the next audio file                    
                    data = wf.readframes(buffer_size) # Read the next audio file into same buffer
                    current_audio_chunk = 0
                    #print("read new "+audio_file+" file. continue playback")
                    if (audio_is_behind):
                        stream.write(data)
                        if (buffer_size<=0):
                            buffer_size = 1
                        data = wf.readframes(int(buffer_size/2+sync_audio_delta_bytes))
                    
                    # check new video
                    videos_played_n+=1
                    #print("end of "+video_file+", searching for next video "+str(current_chunk))
                    video_file = 'modules/wav2lip/temp/result_'+str(current_chunk)+'.mp4'
                    if (not os.path.isfile(video_file)):
                        print("Notice: "+video_file+" is not found, nothing to play as a video")
                        cap.release()
                        #audio_file = "out_"+str(current_chunk)+".wav"
                        #break
                    else:
                        #print (video_file + " and " + audio_file + " exist")
                        cap.release()
                        cap = cv2.VideoCapture(video_file)
                        start_time = time.time()
                        #print ("new "+video_file + " is opened")
            
                    current_chunk+=1
                else:
                    # No more audio files, breaking the loop
                    break
            else:
                # Read the next chunk of audio data into the buffer
                data = wf.readframes(buffer_size)               
            
            current_audio_chunk+=1
            player_loop_running = time.time()
            
        # Close the PyAudio stream
        stream.stop_stream()
        stream.close()

        
        wf.close() # Close the current wav file        
        #pyaudio_p.terminate() # Terminate PyAudio
        #print("exited while")
        #time.sleep(10)
        cap.release()
        #cv2.destroyAllWindows()    # window
    
    return current_chunk
