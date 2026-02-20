from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("file", "Upload CSV or Excel", accept=[".csv", ".xlsx"]),
        ui.output_ui("sheet_selector"),
        ui.output_ui("column_selectors")  # dynamic dropdowns
        #ui.input_select("xcol", "X-axis", choices=cols, selected="Morale"),
        #ui.input_select("ycol", "Y-axis", choices=cols, selected="Stress"),
        #ui.input_select("zcol", "Z-axis", choices=cols, selected=cols[0]),
    ),
    ui.card(
        ui.card_header("Auto Plot"),
        ui.output_plot("plot")
    )
)


def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)

def server(input, output, session):

    # ---- Detect sheet names if Excel ----
    @reactive.calc
    def sheet_names():
        file = input.file()
        if file is None:
            return None

        path = file[0]["datapath"]
        ext = os.path.splitext(file[0]["name"])[1].lower()

        if ext == ".xlsx":
            xls = pd.ExcelFile(path)
            return xls.sheet_names
        return None

    # ---- Sheet selector UI ----
    @output
    @render.ui
    def sheet_selector():
        sheets = sheet_names()
        if sheets is None:
            return None

        return ui.input_select(
            "sheet",
            "Select Sheet",
            choices=sheets,
            selected=sheets[0]
        )

    # ---- Load dataframe reactively ----
    @reactive.calc
    def df():
        file = input.file()
        if file is None:
            return None

        path = file[0]["datapath"]
        ext = os.path.splitext(file[0]["name"])[1].lower()

        if ext == ".csv":
            df = pd.read_csv(path)

        elif ext == ".xlsx":
            sheet = input.sheet()
            if sheet is None:
                return None
            df = pd.read_excel(path, sheet_name=sheet)

        else:
            return None

        df.insert(df.shape[1],"None",np.ones(df.shape[0]))

        #cols = df.columns.tolist()

        #num_data_starts = cols.index("Total Complete Respondents")
        #subset = cols[num_data_starts:]

        #df[subset] = (
        #    df[subset]
        #    .replace(r"^\s*NR\s*$", np.nan, regex=True)  # catches spaces
        #)

        df = df.replace(r"^\s*NR\s*$", np.nan, regex=True)

        # Clean common NR values
        #df = df.replace(r"^\s*NR\s*$", np.nan, regex=True)

        return df
    
    # Dynamic dropdowns once file is loaded
    @output
    @render.ui
    def column_selectors():
        data = df()
        if data is None:
            return ui.p("Upload a CSV to begin.")

        cols = data.columns.tolist()

        return ui.TagList(
            ui.input_select("xcol", "X-axis", choices=cols, selected="None"),
            ui.input_select("ycol", "Y-axis", choices=cols, selected="None"),
            ui.input_select("zcol", "Hue", choices=cols, selected="None")
        )

    @output
    @render.plot
    def plot():

        data = df()
        if data is None:
            return
    
        x = input.xcol()
        y = input.ycol()
        z = input.zcol()

        fig, ax = plt.subplots()

        if (x == "None") and (y == "None") and (z == "None"):
            return fig

        x_num = is_numeric(data[x])
        y_num = is_numeric(data[y])
        z_num = is_numeric(data[z])

        if z_num:
            data[z] = np.where(data[z] > np.nanmedian(data[z]), f"> {np.round(np.nanmedian(data[z]),2)}", f"< {np.round(np.nanmedian(data[z]),2)}")



        if sum([x == "None",y == "None", z == "None"])==2:

            sns.histplot(data=data,x = [i for i in [x,y,z] if i != "None"][0], ax=ax)
        
        # --- Both categorical → count plot ---
        elif not x_num and not y_num:
            sns.countplot(data=data, x=x, hue=y, ax=ax, palette=z)

        # --- One numeric, one categorical → boxplot ---
        elif x_num and not y_num:
            sns.boxplot(data=data, x=y, y=x, ax=ax, hue=z)

        elif not x_num and y_num:
            sns.boxplot(data=data, x=x, y=y, ax=ax, hue=z)
        # --- Both numeric → scatterplot ---
        else:

            #ax.scatter(df[x], df[y], alpha=0.7)

            g = sns.JointGrid(data=data, x=x, y=y,hue=z)

            # Plot using axes-level functions, passing the specific axes
            #sns.scatterplot(data=df, x=x, y=y, ax=g.ax_joint, hue=z, alpha=0.6, marker=".")
            #sns.regplot(data=df, x=x, y=y, ax=g.ax_joint, hue=z, alpha=0.6, marker=".")

            for cat in data[z].unique():
                sns.regplot(data=data.loc[data[z]==cat], x=x, y=y, ax=g.ax_joint, marker=".", label=cat)

            if z != "None":
                g.ax_joint.legend()
                g.ax_joint.get_legend().set_title(z)

            sns.histplot(data=data, x=x, ax=g.ax_marg_x, hue=z, legend=False)
            sns.histplot(data=data, y=y, ax=g.ax_marg_y, hue=z, legend=False)

            g.ax_joint.set_xlabel(x)
            g.ax_joint.set_ylabel(y)
            #g.ax_joint.set_title(f"{y} vs {x}")

            # Better title placement
            g.ax_joint.set_title(f"{y} vs {x}", pad=45)

            # Fix label spacing
            g.ax_joint.set_xlabel(x, labelpad=15)
            g.ax_joint.set_ylabel(y, labelpad=15)

            g.figure.subplots_adjust(
                left=0.15,
                bottom=0.15,
                top=.9
            )

            #plt.suptitle(f"{y} vs {x}", y=1)
            #g.plot_joint(sns.scatterplot, alpha=0.6)
            g.plot_marginals(sns.histplot, kde=True)

            return g.figure



        if sum([x == "None",y == "None", z == "None"])==2:
            ax.set_title(f"{[i for i in [x,y,z] if i != "None"][0]}")
            ax.set_xlabel([i for i in [x,y,z] if i != "None"][0])
            ax.set_ylabel("Frequency")  
        else:
            ax.set_title(f"{y} vs {x}")
            ax.set_xlabel(x)
            ax.set_ylabel(y)   
        
        if ax.get_legend() is not None:
            if z == "None":
                ax.get_legend().remove()

        plt.tight_layout()
        return fig

app = App(app_ui, server)


