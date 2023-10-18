# Import the EvaDB package
import evadb
from video_analysis import ObjectDetector

# Connect to EvaDB and get a database cursor for running queries
cursor = evadb.connect().cursor()

cursor.query("""
    CREATE FUNCTION IF NOT EXISTS ObjectDetector
    INPUT  (video_path TEXT(ANYDIM))
    OUTPUT (output_video_path TEXT(ANYDIM))
    TYPE  Object_detection
    IMPL  'video_analysis.py';
    """).df()
print(cursor.query("SHOW FUNCTIONS;").df())

cursor.query("DROP TABLE IF EXISTS video_info").df()
cursor.query("""CREATE TABLE IF NOT EXISTS video_info(
             video_id INTEGER,
             video_name TEXT(ANYDIM),
             video_path TEXT(ANYDIM));
             """).df()
cursor.query("""
    INSERT INTO video_info(video_id, video_name, video_path)
    VALUES (1,"ringing doorbell", "/workspace/frigate/media/ringing_doorbell.mp4"
    );
""").df()

result=cursor.query("""
             SELECT video_name,video_id,video_path FROM video_info;
            """).df()
print(result)

result=cursor.query("SELECT video_id, ObjectDetector(video_path) FROM video_info").df()
print(result)
cursor.query("DROP FUNCTION ObjectDetector").df()


