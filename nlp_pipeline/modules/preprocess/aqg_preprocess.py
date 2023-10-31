import re


def remove_viquad_noise(context: str) -> str:
    start = re.search(r"\[\b", context)
    end = re.search(r"\b\]", context)
    if (start and end) is not None:
        start = start.start()
        end = end.end()
        context = context.replace(context[start:end], "")
        return remove_viquad_noise(context)
    else:
        return context
