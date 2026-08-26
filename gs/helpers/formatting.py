def format_number(n):
    """
    Format a number so that it reads approximate stats: 1-999, 1.Xk-999.Xk, 1.XM-999.XM, 1.XB-999.XB, etc.
    """
    if n < 1000:
        return str(n)
    elif n < 1000000:
        return f"{n / 1000:.1f}k"
    elif n < 1000000000:
        return f"{n / 1000000:.1f}M"
    else:
        return f"{n / 1000000000:.1f}B"
