from manim import *; from manim.utils import tex_to_string

class OpeningQuestion(Scene):
    def construct(self):
        question = Text("What if a computer could learn from experience?").scale(1.5)
        self.play(FadeIn(question))

    def construct(self):
        audio = AudioFileSource("audio.mp3")
        visual = ImageMobject("human_brain.png", "computer_icon.png")
        caption = Text("Learning from experience?").scale(1.5)
        self.play(FadeIn(audio), FadeIn(visual), FadeIn(caption))

    def construct(self):
        introduction = Text("Supervised learning is a method that allows computers to do just that...").scale(1.5)
        image = ImageMobject("child_teaching.png", "labeled_data.png")
        self.play(FadeIn(introduction), FadeIn(image))