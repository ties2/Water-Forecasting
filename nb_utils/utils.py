import inspect


# def get_source_without_docs(func: callable) -> str:
#     source = inspect.getsource(func)
#     source_doc = inspect.getdoc(func)
#     if source_doc is None:
#         return source
#
#     doc_lines = source_doc.split('\n')
#     for l in doc_lines:
#         source = source.replace(l, '')
#     source = source.replace("\"\"\"", '')  # remove docstring quotes
#     return source


def fn_unindent_multiline_string(code: str) -> list[str]:
    orig_lines = code.split('\n')

    # check if this seems to be a multiline string (i.e. first and last lines are empty)
    if not (len(orig_lines) > 2 and len(orig_lines[0].lstrip()) == 0 and len(orig_lines[-1].lstrip()) == 0):
        # not properly formatted multiline, return as-is
        return orig_lines

    # replace tabs by spaces
    lines = [line.replace('\t', '    ') for line in orig_lines]

    # find the lowest indentation of content lines
    min_indent = 9999
    for line in lines[1:-1]:
        stripped_line = line.lstrip()
        if len(stripped_line) == 0:
            continue  # ignore empty lines
        indent_count = len(line) - len(stripped_line)
        min_indent = min(min_indent, indent_count)

    # find indentation of last line - due to checks above this should be equal to the length of the line
    last_indent = len(lines[-1])

    if min_indent < last_indent:
        # some lines seem to have negative indentation, so improperly formatted multiline string. Return as-is.
        return orig_lines

    # return unindented lines
    return [line[last_indent:] for line in lines[1:-1]]


def get_one_or_many_string(string: list[str] | str | None, unindent_multiline_string: bool = True, none_as_list: bool = True) -> list[str]:
    if string is None:
        return [] if none_as_list else None

    if isinstance(string, str):
        if unindent_multiline_string:
            string = fn_unindent_multiline_string(string)
        else:
            string = [string]

    return string


def get_source_without_docs(func: callable) -> str:
    # get source, with tabs replaced by spaces
    source = inspect.getsource(func)
    source = source.replace('\t', '    ')

    # check if function actually has docstring
    source_doc = inspect.getdoc(func)
    if source_doc is None:
        # no docs to clean-up, return source as-is
        return source

    # find docstring quotes
    source_lines = source.split('\n')
    doc_start = -1
    doc_stop = -1
    for i, line in enumerate(source_lines):
        quotes_pos = line.find('"""')
        if quotes_pos != -1:
            if doc_start == -1:
                doc_start = i
            else:
                doc_stop = i
                break

    # docstring found: remove all lines
    if doc_start > -1 and doc_stop > -1:
        valid_lines = source_lines[:doc_start]
        valid_lines.extend(source_lines[doc_stop + 1:])
        return '\n'.join(valid_lines)

    # fallback to trimming documentation lines in source directly
    doc_lines = source_doc.split('\n')
    for l in doc_lines:
        source = source.replace(l, '')
    source = source.replace("\"\"\"", '')  # remove docstring quotes
    return source


def get_func_body(func: callable, remove_doc_string: bool = True) -> str:
    """
    Retrieve body of given function as a string.
    Removes function header and return statement.
    Docstring removal is optional.

    :param func: function that should be processed
    :param remove_doc_string: if enabled also removes the docstring from the function.
    :return: body of the function as a string
    """
    if remove_doc_string:
        source = get_source_without_docs(func)
    else:
        source = inspect.getsource(func)

    # remove header of function
    header_end = 0
    parentheses_cnt = 0
    for idx, c in enumerate(source):
        parentheses_cnt += 1 if c == '(' else 0
        parentheses_cnt -= 1 if c == ')' else 0
        colon_found = True if parentheses_cnt == 0 and source[idx] == ':' else False

        if parentheses_cnt == 0 and colon_found:
            header_end = idx + 1
            break
    source = source[header_end:]  # remove header from source
    source = source.strip('\n ')  # remove trailing and leading newlines
    source = "    " + source  # re-add indentation of first line :(
    source_lines = source.split('\n')  # string to list of lines
    source_lines = [l[4:] for l in source_lines]  # strip indents of body

    # remove all lines starting from the last return statement
    lines_to_delete = 0
    for line in reversed(source_lines):
        lines_to_delete += 1
        if 'return' in line:
            break

    # still want to keep functions that do not have a return
    if lines_to_delete < len(source_lines):
        source_lines = source_lines[:-lines_to_delete]

    return "\n".join(source_lines)
