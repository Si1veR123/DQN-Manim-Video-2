import manim
import numpy as np


def line_centerer(text, width=17, font_size=30):
    """
    Splits and centers a single line of text to multiple lines based on width.
    """
    text_group = manim.VGroup()
    newest_line = []
    # add new lines automatically
    for word in text.split(" "):
        current_line_text_obj = manim.Text(" ".join(newest_line + [word]))
        if current_line_text_obj.width > width:
            new_line_text = manim.Text(" ".join(newest_line), font_size=font_size)
            text_group.add(new_line_text)

            newest_line = [word]
        else:
            newest_line.append(word)
    new_line_text = manim.Text(" ".join(newest_line), font_size=font_size)
    text_group.add(new_line_text)
    text_group.arrange(manim.DOWN, center=True)
    return text_group


class CodeExplained:
    def __init__(self, scene: manim.Scene, code: str, explained: str):
        self.scene = scene

        self.code = manim.Code(
            language="Python",
            style="one-dark",
            font="consolas",
            background_stroke_color=manim.ORANGE,
            code=code
        )

        self.code.height = 4
        if self.code.width > 12:
            self.code.width = 12
        self.code.set_y(1.5)

        self.lines = len(code.splitlines())

        self.explained = line_centerer(explained)
        self.explained.set_y(-1.65)

        self.progress_line = manim.Line(np.array((-7.1, -4, 0)),
                                        np.array((-7.1, -4, 0)),
                                        stroke_width=13,
                                        stroke_color=manim.ORANGE)

    def play_anims(self, forward=True):
        anims = self.get_anims(forward=forward)
        self.scene.play(*anims)
        if forward:
            self.progress_line.generate_target(use_deepcopy=True)
            # move target end to right side of screen
            self.progress_line.target.put_start_and_end_on(self.progress_line.start, np.array((7.1, -4, 0)))
            length = 4 + 0.5*self.lines
            self.scene.play(manim.MoveToTarget(self.progress_line,
                                               run_time=length,
                                               rate_func=manim.rate_functions.ease_in_out_sine))

    def get_anims(self, forward=True):
        if forward:
            return manim.Write(self.code), manim.Write(self.explained), manim.Write(self.progress_line)
        else:
            return manim.ScaleInPlace(self.code, 0), manim.Unwrite(self.explained), manim.Unwrite(self.progress_line)
