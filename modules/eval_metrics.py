import numpy as np
import wandb
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix


def get_static_confusion_matrix(confmatrix,classes):
    z = confmatrix
    x = classes
    y = classes
    

    z_text = [[str(y) for y in x] for x in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    # fig.update_layout(title_text='',
    #                   #xaxis = dict(title='x'),
    #                   yaxis = dict(title='Predicted')
    #                  )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=13),
                            x=0.5,
                            y=-0.34,
                            showarrow=False,
                            text="<b>True Class</b>",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=13),
                            x=-0.2,
                            y=0.5,
                            showarrow=False,
                            text="<b>Predicted</b>",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title

    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    fig['layout']['xaxis']['side'] = 'bottom'
    return fig


def get_confusion_matrix(y_pred, y_val, n_classes, labels):
    
    confmatrix = confusion_matrix(y_pred, y_val, labels=range(n_classes)) 

    saved_confmatrix = np.copy(confmatrix).tolist()
    static_fig = get_static_confusion_matrix(saved_confmatrix,labels)


    confdiag = np.eye(len(confmatrix)) * confmatrix 
    np.fill_diagonal(confmatrix, 0)
    confmatrix = confmatrix.astype('float')
    n_confused = np.sum(confmatrix)


    confmatrix[confmatrix == 0] = np.nan
    confmatrix = go.Heatmap({'coloraxis': 'coloraxis1', 'x': labels, 'y': labels, 'z': confmatrix,
    'hoverongaps':False, 'hovertemplate': 'Predicted %{y}<br>Instead of %{x}<br>On %{z} examples<extra></extra>'})

    confdiag = confdiag.astype('float')
    n_right = np.sum(confdiag)
    confdiag[confdiag == 0] = np.nan
    confdiag = go.Heatmap({'coloraxis': 'coloraxis2', 'x': labels, 'y': labels, 'z': confdiag,
    'hoverongaps':False, 'hovertemplate': 'Predicted %{y} just right<br>On %{z} examples<extra></extra>'})


    fig = go.Figure((confdiag, confmatrix))
    transparent = 'rgba(0, 0, 0, 0)'
    n_total = n_right + n_confused

    fig.update_layout({'coloraxis1': {'colorscale': [[0, transparent], [0, 'rgba (180, 0, 0, 0.05)'], [1, f'rgba (180, 0, 0, {max (0.2, (n_confused/n_total)**0.5)})']],'showscale': False}})
    fig.update_layout({'coloraxis2': {'colorscale': [[0, transparent], [0, f'rgba(0, 180, 0, { min(0.8, (n_right/n_total) ** 2)})'], [1, 'rgba(0, 180, 0, 1) ']],'showscale': False}})

    xaxis = { 'title': {'text': 'y_true'},'showticklabels': False} 
    yaxis = { 'title': {'text': 'y_pred'},'showticklabels': False}


    fig.update_layout(title={'text': 'Confusion matrix', 'x': 0.5}, paper_bgcolor=transparent, plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)

    return {'confusion_matrix': wandb.data_types.Plotly(fig), 'static_confusion_matrix': wandb.data_types.Plotly(static_fig)}, saved_confmatrix
