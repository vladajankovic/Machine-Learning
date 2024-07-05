import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import pandas as pd

from machine_learning import LinearRegressionGradientDescent
from machine_learning import LogisticRegressionOneVsOne
from machine_learning import LogisticRegressionMultinomial


class GUIApp:
    def __init__(self,
                 master,
                 data: pd.DataFrame,
                 cat_enc_dict: dict,
                 pub_enc_dict: dict,
                 linreg: LinearRegressionGradientDescent,
                 logreg_ovo: LogisticRegressionOneVsOne,
                 logreg_multi: LogisticRegressionMultinomial,
                 price_cat_dict: dict
                 ):
        self.master = master
        self.data = data
        self.cat_enc_dict = cat_enc_dict
        self.pub_enc_dict = pub_enc_dict
        self.linreg = linreg
        self.logreg_ovo = logreg_ovo
        self.logreg_multi = logreg_multi
        self.price_cat_dict = price_cat_dict

        master.title("Predvidjanje cene knjiga")
        master.geometry("500x400")

        main_frame = tk.Frame(master, padx=20, pady=20)
        main_frame.pack(expand=True)

        # Publishers dropdown
        tk.Label(main_frame, text="Izdavac:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.publishers = sorted(list(self.data['izdavac'].unique()))
        self.publisher_var = tk.StringVar()
        self.publisher_dropdown = ttk.Combobox(main_frame, textvariable=self.publisher_var, values=self.publishers, width=40)
        self.publisher_dropdown.grid(row=0, column=1, padx=5, pady=5)
        self.publisher_dropdown.set("Izaberite opciju")

        # Categories dropdown
        tk.Label(main_frame, text="Kategorija:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.categories = sorted(list(self.data['kategorija'].unique()))
        self.category_var = tk.StringVar()
        self.category_dropdown = ttk.Combobox(main_frame, textvariable=self.category_var, values=self.categories, width=40)
        self.category_dropdown.grid(row=1, column=1, padx=5, pady=5)
        self.category_dropdown.set("Izaberite opciju")

        # Number of pages slider
        tk.Label(main_frame, text="Broj strana:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.pages_var = tk.IntVar()
        self.pages_slider = tk.Scale(main_frame, from_=6, to=1400, orient=tk.HORIZONTAL, variable=self.pages_var, length=200)
        self.pages_slider.grid(row=2, column=1, padx=5, pady=5)

        # Year of publishing spinbox
        tk.Label(main_frame, text="Godina izdavanja:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.year_var = tk.IntVar()
        self.year_spinbox = tk.Scale(main_frame, from_=2005, to=2024, orient=tk.HORIZONTAL, variable=self.year_var, length=200)
        self.year_spinbox.grid(row=3, column=1, padx=5, pady=5)

        # Format dropdown
        tk.Label(main_frame, text="Format:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        self.formats = sorted(list(self.data['format'].unique()))
        self.format_var = tk.StringVar()
        self.format_dropdown = ttk.Combobox(main_frame, textvariable=self.format_var, values=self.formats)
        self.format_dropdown.grid(row=4, column=1, padx=5, pady=5)
        self.format_dropdown.set("Izaberite opciju")

        # Binding radio buttons
        tk.Label(main_frame, text="Tip poveza:").grid(row=5, column=0, sticky="e", padx=5, pady=5)
        self.binding_var = tk.StringVar()
        self.binding_var.set("Tvrd")
        tk.Radiobutton(main_frame, text="Tvrd", variable=self.binding_var, value="Tvrd").grid(row=5, column=1, sticky="w", padx=5)
        tk.Radiobutton(main_frame, text="Broš", variable=self.binding_var, value="Broš").grid(row=5, column=1, sticky="e", padx=5)

        tk.Label(main_frame, text="Model predviđanja:").grid(row=6, column=0, sticky="e", padx=5, pady=5)
        self.model_var = tk.StringVar()
        self.model_var.set("Linarna regresija")
        tk.Radiobutton(main_frame, text="Linarna regresija",
                       variable=self.model_var, value="Linarna regresija"
                       ).grid(row=6, column=1, sticky="w", padx=5)
        tk.Radiobutton(main_frame, text="Logistička regresija - jedan nasuprot jednom",
                       variable=self.model_var, value="Logistička regresija - jedan nasuprot jednom"
                       ).grid(row=7, column=1, sticky="w", padx=5)
        tk.Radiobutton(main_frame, text="Logistička regresija - multinomijalna",
                       variable=self.model_var, value="Logistička regresija - multinomijalna"
                       ).grid(row=8, column=1, sticky="w", padx=5)


        # Submit button
        submit_button = tk.Button(main_frame, text="Predvidi cenu", command=self.submit)
        submit_button.grid(row=9, column=0, columnspan=2, pady=20)

    def submit(self):
        publisher = self.publisher_var.get()
        category = self.category_var.get()
        pages = self.pages_var.get()
        year = self.year_var.get()
        format_type = self.format_var.get()
        binding = self.binding_var.get()
        model = self.model_var.get()

        if publisher != "Izaberite opciju" and category != "Izaberite opciju" and format_type != "Izaberite opciju":

            fmt = [float(v) for v in format_type.split('x')]
            surface = fmt[0] * fmt[1]
            min_surface = min(self.data['povrsina'].to_numpy())
            max_surface = max(self.data['povrsina'].to_numpy())

            d = {
                'broj_strana': [pages / 100],
                'godina_izdavanja': [year % 100],
                'povrsina': [(surface - min_surface) / (max_surface - min_surface)],
                'tip_poveza': [1 if binding == 'Tvrd' else 0],
                'kategorija': [self.cat_enc_dict[category] / 100],
                'izdavac': [self.pub_enc_dict[publisher] / 100]
            }

            X = pd.DataFrame(data=d)

            print(X)

            price = '0.00 din'

            if model == 'Linarna regresija':
                price = self.linreg.predict(X)[0] * 100
                price = f'{price:.2f} din'
            if model == 'Logistička regresija - jedan nasuprot jednom':
                price = self.logreg_ovo.predict(X)[0]
                price = self.price_cat_dict[price]
            if model == 'Logistička regresija - multinomijalna':
                price = self.logreg_multi.predict(X)[0]
                price = self.price_cat_dict[price]

            message = (f"Submitted:\n"
                       f"Izdavac: {publisher}\n"
                       f"Kategorija: {category}\n"
                       f"Broj strana: {pages}\n"
                       f"Godina izdavanja: {year}\n"
                       f"Format: {format_type}\n"
                       f"Povez: {binding}\n"
                       f"Cena: {price}")
            messagebox.showinfo("Submission", message)
        else:
            messagebox.showerror("Error", "Niste popunili sva polja")


def create_app(
        data: pd.DataFrame,
        cat_enc_dict: dict,
        pub_enc_dict: dict,
        linreg: LinearRegressionGradientDescent,
        logreg_ovo: LogisticRegressionOneVsOne,
        logreg_multi: LogisticRegressionMultinomial,
        price_cat_dict: dict
):
    root = tk.Tk()
    app = GUIApp(
        root,
        data,
        cat_enc_dict,
        pub_enc_dict,
        linreg,
        logreg_ovo,
        logreg_multi,
        price_cat_dict
    )
    root.mainloop()