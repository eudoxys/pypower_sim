"""KML generator

The KML class collect KML entities and saves them to a file when the `close
()` method is called. Note that the KML file is not saved if the KML object
is deleted without being closed.

The following methods are used to create KML files:

- `pypower_sim.kml.KML.add_linestyle` creates a linestyle for use by line entities

- `pypower_sim.kml.KML.add_markerstyle` creates a markerstyle for use by market entities

- `pypower_sim.kml.KML.add_line` creates a line entity

- `pypower_sim.kml.KML.add_marker` creates a marker entity

# See also

- [Google Earth KML Documentation](https://developers.google.com/kml/documentation)
"""

class KML:
    """KML generator class"""

    def __init__(self,
        kmlfile:str,
        name:str=None
        ):
        """Construct a KML file generator

        # Arguments

        - `kmlfile`: KML file name

        - `name`: KML name (if different from base name `kmlfile`)
        """

        self.kmlfile = kmlfile
        """KML file name"""

        self.name = name if name else os.path.splitext(os.path.basename(self.kmlfile))[0]
        """KML name (if different from base name of `pypower_sim.kml.KML.kmlfile`)"""
        
        self.line = {}
        """Line entity table (see `pypower_sim.kml.KML.add_line`)"""

        self.linestyle = {}
        """Line style table (see `pypower_sim.kml.KML.add_linestyle`)"""

        self.marker = {}
        """Place marker entity table (see `pypower_sim.kml.KML.add_marker`)"""

        self.markerstyle = {}
        """Place marker style table (see `pypower_sim.kml.KML.add_markerstyle`)"""

        self.folders = {}
        """@private Folder table"""

    def __del__(self):
        if self.kmlfile:
            self.close()

    def add_linestyle(self,name:str,**kwargs):
        """Add a line style

        # Arguments:

        - `name`: linestyle name

        - `color`: line color

        - `width`: line width

        - `opacity`: line opacity
        """
        self.linestyle[name] = kwargs

    def add_markerstyle(self,name:str,**kwargs):
        """Add a marker style

        # Arguments

        - `name`: markerstyle name

        - `icon`: icon URL

        - `scale`: icon size
        """
        self.markerstyle[name] = kwargs

    def add_line(self,name:str,**kwargs):
        """Add a line entity

        # Arguments

        - `name`: line name

        - `from_position`: line starting position

        - `to_position`: line ending position

        - `style`: line style

        - `data`: line data
        """
        self.line[name] = kwargs

    def add_marker(self,name:str,**kwargs):
        """Add a marker entity

        # Arguments

        - `name`: marker name

        - `position`: marker position

        - `style`: marker style

        - `data`: marker data
        """
        self.marker[name] = kwargs

    def add_folder(self,name:str,**kwargs):
        """@private Add a folder

        # Arguments

        - `name`: folder name

        - `parent`: parent folder name
        """
        self.folder[name] = kwargs

    def close(self,write=True):
        """Close KML file

        # Arguments

        - `write`: enable output to `pypower_sim.kml.KML.kmlfile`
        """
        if self.kmlfile and write:
            with open(self.kmlfile,"w",encoding="utf-8") as fh:

                print('<?xml version="1.0" encoding="UTF-8"?>',file=fh)
                print('<kml xmlns="http://www.opengis.net/kml/2.2"'
                    ' xmlns:gx="http://www.google.com/kml/ext/2.2">',file=fh)
                print("<Document>",file=fh)
                print(f"  <name>{self.name}</name>",file=fh)

                # output marker styles
                for name,data in self.markerstyle.items():
                    print(f'  <Style id="{name}">',file=fh)
                    print("    <IconStyle>",file=fh)
                    if "scale" in data:
                        print(f"      <scale>{data['scale']}</scale>",file=fh)
                    print("      <Icon>",file=fh)
                    print(f"        <href>{data['url']}</href>",file=fh)
                    print("      </Icon>",file=fh)
                    print("    </IconStyle>",file=fh)
                    print("  </Style>",file=fh)


                # output line styles
                for name,data in self.linestyle.items():
                    print(f'  <Style id="{name}">',file=fh)
                    print("    <PolyStyle>",file=fh)
                    print(f"      <color>{data['color']}</color>",file=fh)
                    print("    </PolyStyle>",file=fh)
                    print("    <LineStyle>",file=fh)
                    print(f"      <color>{data['color']}</color>",file=fh)
                    print(f"      <width>{data['width']}</width>",file=fh)
                    print("    </LineStyle>",file=fh)
                    print("  </Style>",file=fh)

                # output markers
                for name,data in self.marker.items():
                    print("  <Placemark>",file=fh)
                    if name:
                        print(f"    <name>{name}</name>""",file=fh)
                    if "style" in data:
                        print(f"    <styleUrl>#{data['style']}</styleUrl>",file=fh)
                    print(f"""    <Point><coordinates>{','.join(f'{x}'
                        for x in data['position'])}</coordinates></Point>""",file=fh)
                    print("  </Placemark>",file=fh)

                # output lines
                for name,data in self.line.items():
                    print("  <Placemark>",file=fh)
                    if name:
                        print(f"    <name>{name}</name>",file=fh)
                    if "style" in data:
                        print(f"    <styleUrl>#{data['style']}</styleUrl>",file=fh)
                    print("    <LineString>",file=fh)
                    print("      <tesselate>1</tesselate>",file=fh)
                    print("      <coordinates>",file=fh)
                    print(f"""      {','.join(f'{x}'
                        for x in data['from_position'])}""",file=fh)
                    print(f"""      {','.join(f'{x}'
                        for x in data['to_position'])}""",file=fh)
                    print("      </coordinates>",file=fh)
                    print("    </LineString>",file=fh)
                    print("  </Placemark>",file=fh)
                print("</Document>",file=fh)
                print("</kml>""",file=fh)

        self.kmlfile = None
