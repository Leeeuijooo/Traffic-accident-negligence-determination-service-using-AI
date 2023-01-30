# 유튜브 영상 따오기

from pytube import YouTube
YouTube('https://www.youtube.com/watch?v=boViWJsxtu8').streams.get_highest_resolution().download()

# 영상 자르기는 '트리밍' 기능 이용해서 자르기