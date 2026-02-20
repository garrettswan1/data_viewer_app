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
    ),
    ui.card(
        ui.card_header("Auto Plot"),
        ui.output_plot("plot"),
        ui.output_text_verbatim("text")
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
        if file is None: # need this if check given that first column selected is None
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

        # this probably needs some changes
        df = df.replace(r"^\s*NR\s*$", np.nan, regex=True)

        return df
    
    # Dynamic dropdowns once file is loaded
    @output
    @render.ui
    def column_selectors():
        source = df()
        if source is None:
            return ui.p("Upload a CSV to begin.")
        
        data = source.copy(deep=True)

        cols = data.columns.tolist()

        # default option should be None
        return ui.TagList(
            ui.input_select("xcol", "X-axis", choices=cols, selected="None"),
            ui.input_select("ycol", "Y-axis", choices=cols, selected="None"),
            ui.input_select("zcol", "Hue", choices=cols, selected="None")
        )

    @output
    @render.plot
    def plot():

        source = df()

        if source is None:
            return
        
        data = source.copy(deep=True)

        x = input.xcol()
        y = input.ycol()
        z = input.zcol()

        fig, ax = plt.subplots()

        if (x == "None") and (y == "None") and (z == "None"):
            return fig
        
        x_num = is_numeric(data[x])
        y_num = is_numeric(data[y])
        z_num = is_numeric(data[z])

        tmp_data = pd.DataFrame({x: data[x], y: data[y]})

        if z_num:

            median = np.nanmedian(data[z])

            hue_name = f"{z}_split"

            tmp_data[hue_name] = np.where(
                data[z] > median,
                f"> {median:.2f}",
                f"≤ {median:.2f}"
            )

            tmp_data[hue_name] = pd.Categorical(
                tmp_data[hue_name],
                categories=[f"≤ {median:.2f}", f"> {median:.2f}"],
                ordered=True
            )

        else:

            hue_name = z
            tmp_data[hue_name] = data[z]

            


        if sum([x == "None", y == "None"])==1:
            # if just one

            sns.histplot(data=tmp_data,x = [i for i in [x,y] if i != "None"][0], hue=hue_name, ax=ax)                

        else:

            # --- Both categorical → count plot ---
            if not x_num and not y_num:
                #sns.countplot(data=data, x=x, hue=y, ax=ax, palette=z)

                contingency_table  = pd.crosstab(tmp_data[x], [tmp_data[y], tmp_data[hue_name]], rownames=[x], colnames=[y, hue_name])
                sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu", ax=ax)

            # --- One numeric, one categorical → boxplot ---
            elif (x_num and not y_num) or (not x_num and y_num):
                sns.boxplot(data=tmp_data, x=x, y=y, ax=ax, hue=hue_name)

            # --- Both numeric → scatterplot ---
            else:

                #ax.scatter(df[x], df[y], alpha=0.7)

                g = sns.JointGrid(data=tmp_data, x=x, y=y,hue=hue_name)

                # Plot using axes-level functions, passing the specific axes
                #sns.scatterplot(data=df, x=x, y=y, ax=g.ax_joint, hue=z, alpha=0.6, marker=".")
                #sns.regplot(data=df, x=x, y=y, ax=g.ax_joint, hue=z, alpha=0.6, marker=".")

                for cat in tmp_data[hue_name].unique():
                    sns.regplot(data=tmp_data.loc[tmp_data[hue_name]==cat], x=x, y=y, ax=g.ax_joint, marker=".", label=cat)

                if hue_name != "None":
                    g.ax_joint.legend()
                    g.ax_joint.get_legend().set_title(hue_name)

                sns.histplot(data=tmp_data, x=x, ax=g.ax_marg_x, hue=hue_name, legend=False)
                sns.histplot(data=tmp_data, y=y, ax=g.ax_marg_y, hue=hue_name, legend=False)

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

        if sum([x == "None",y == "None"])==1:
            ax.set_title(f"{[i for i in [x,y] if i != "None"][0]}")
            ax.set_xlabel([i for i in [x,y] if i != "None"][0])
            ax.set_ylabel("Frequency")  
        else:
            ax.set_title(f"{y} vs {x}")
            ax.set_xlabel(x)
            ax.set_ylabel(y)   
        
        if ax.get_legend() is not None:
            if hue_name == "None":
                ax.get_legend().remove()

        plt.tight_layout()

        return fig

    @render.text
    def text():

        source = df()

        if source is None:
            return
        
        data = source.copy(deep=True)

        x = input.xcol()
        y = input.ycol()
        z = input.zcol()

        if (x == "None") and (y == "None") and (z == "None"):
            return "No Variables Selected"

        x_num = is_numeric(data[x])
        y_num = is_numeric(data[y])
        z_num = is_numeric(data[z])

        tmp_data = pd.DataFrame({x: data[x], y: data[y]})

        if z_num:

            median = np.nanmedian(data[z])

            hue_name = f"{z}_split"

            tmp_data[hue_name] = np.where(
                data[z] > median,
                f"> {median:.2f}",
                f"≤ {median:.2f}"
            )

            tmp_data[hue_name] = pd.Categorical(
                tmp_data[hue_name],
                categories=[f"≤ {median:.2f}", f"> {median:.2f}"],
                ordered=True
            )

        else:

            hue_name = z
            tmp_data[hue_name] = data[z]

        #else:
        return_str = ""
        for cat in tmp_data[hue_name].unique():

            return_str += cat + "\n" + tmp_data.loc[tmp_data[hue_name]==cat].describe(include='all').to_string() + "\n ------ \n"
        
        return return_str



app = App(app_ui, server)


