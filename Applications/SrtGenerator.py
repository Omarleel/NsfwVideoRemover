class SrtGenerator:
    def __init__(self):
        self.subtitles = []

    def add_subtitle(self, start_time, end_time, text):
        subtitle = {
            'start_time': start_time,
            'end_time': end_time,
            'text': text
        }
        self.subtitles.append(subtitle)

    def format_time(self, time_seconds):
        hours, remainder = divmod(time_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = (time_seconds - int(time_seconds)) * 1000
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"

    def generate_srt(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            for index, subtitle in enumerate(self.subtitles, start=1):
                start_time = self.format_time(subtitle['start_time'])
                end_time = self.format_time(subtitle['end_time'])
                text = subtitle['text']

                file.write(f"{index}\n")
                file.write(f"{start_time} --> {end_time}\n")
                file.write(f"{text}\n\n")
