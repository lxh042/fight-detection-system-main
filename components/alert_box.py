def render_alert_box(color):
    return f"""
    <div style="
        width: 100%;
        height: 250px;
        border: 5px solid {color};
        border-radius: 10px;
        padding: 15px;
        background-color: {color};
        overflow-y: auto;
    ">
        <h4 style='margin-top: 0;'>Incident Alert Box</h4>
        <p>Dynamic incident info or summary will go here.</p>
    </div>
    """
