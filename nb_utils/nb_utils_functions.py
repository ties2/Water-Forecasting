from IPython.display import Markdown as md


class NBLink:
    def __init__(self, ip, topicname, topic_super_class: str = ""):
        self.topicname = topicname
        self.topic_super_class = topic_super_class
        self.ip = ip

    def __call__(self, i, topic_nb_name, descr):
        return f"Exercise {i} can be found by clicking [{descr}](http://{self.ip}/notebooks/{f'/{self.topic_super_class}' if len(self.topic_super_class) > 0 else ''}/{self.topicname}/{topic_nb_name}/{topic_nb_name}.ipynb)"


def nb_init(topic_name: str, topic_super_class: str = ""):
    if topic_name not in NOTEBOOK_DICT:
        raise RuntimeError("Please fill in the correct <topicname>.")
    context = {'topic_name': topic_name, 'topic_super_class': topic_super_class, 'ip': EXERCISES_IP}
    return context


def exercises(context):
    link = NBLink(context['ip'], context['topic_name'], context['topic_super_class'])
    topic_name = context['topic_name']
    keys = NOTEBOOK_DICT[topic_name]

    if len(keys) == 0:
        output = f"There are no exercise notebooks associated with the '**{topic_name}**' topic yet..."
    else:
        output = "## Links to Exercises\n" \
                 f"Found **{len(keys)}** Jupyter Notebooks associated with the '**{topic_name}**' topic:\n"
        for i, key in enumerate(keys):
            lnk = link(i + 1, key, f'{topic_name}/{key}')
            output += f"- {lnk}\n"

    return md(output)


# dummy values
EXERCISES_IP = ""
NOTEBOOK_DICT: dict[str, list] = {}
