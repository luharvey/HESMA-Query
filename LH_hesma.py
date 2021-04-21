#█░█ █▀▀ █▀ █▀▄▀█ ▄▀█
#█▀█ ██▄ ▄█ █░▀░█ █▀█		-Luke Harvey 19/4/2021
"""
This module contains classes and functions to download and analyse data from the Heidelberg Supernova Model Archive (HESMA). A profile is required to access this data, this username and password
will be required to download the data and will be encrypted and stored in .HESMA_credentials.
"""

import matplotlib.pyplot as plt 
import numpy as np
from pylh import rainbow_array
import requests
import glob
from getpass import getpass
from os import getcwd
import base64
from lxml import html

#Text properties
class text:
	PURPLE = '\033[95m'
	CYAN = '\033[96m'
	DARKCYAN = '\033[36m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	RED = '\033[91m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	END = '\033[0m'

creds = glob.glob('./.HESMA_credentials')
if len(creds) == 0:
	print('')
	username = input('HESMA Username: ')
	password = getpass('HESMA Password: ')
	print('')
	root_path = getcwd()

	with open('./.HESMA_credentials', 'w') as file:
		file.write( str(base64.b64encode(username.encode("utf-8"))) + ' \n')
		file.write( str(base64.b64encode(password.encode("utf-8"))) + ' \n')
		file.write(root_path + ' \n')

payload = {}

U_colour = '#DA05F7'
B_colour = '#00CFE3'
V_colour = '#44CF02'
R_colour = '#EB1500'

with open('./.HESMA_credentials', 'r') as file:
	lines = file.readlines()

	un = lines[0].split(sep = ' ')[0]
	payload['u'] = base64.b64decode(un[2:len(un)-1]).decode("utf-8")

	pw = lines[1].split(sep = ' ')[0]
	payload['p'] = base64.b64decode(pw[2:len(pw)-1]).decode("utf-8")

	root_path = lines[2].split(sep = ' ')[0]

def download_HESMA_data(model_name, get_time = False):
	login_url = 'https://hesma.h-its.org/doku.php?id=start&do=login&sectok='

	isotope_check = glob.glob(root_path + '/isotopes/' + model_name + '_isotopes.dat')
	spectra_check = glob.glob(root_path + '/spectra/' + model_name + '_spectra.dat')
	lightcurve_check = glob.glob(root_path + '/lightcurve/' + model_name + '_lightcurve.dat')
	lightcurves_early_check = glob.glob(root_path + '/lightcurves_early/' + model_name + '_lightcurves_early.dat')
	density_check = glob.glob(root_path + '/density/' + model_name + '_density.dat')

	if len(isotope_check) == 0 or len(spectra_check) == 0 or len(lightcurve_check) == 0 or len(lightcurves_early_check) == 0 or len(density_check) == 0:
		#Opening session with login credentials
		with requests.Session() as s:
			p = s.post(login_url, data = payload)
			isotopes_url = 'https://hesma.h-its.org/lib/exe/fetch.php?media=data:models:' + model_name + '_isotopes.dat'
			spectra_url = 'https://hesma.h-its.org/lib/exe/fetch.php?media=data:models:' + model_name + '_spectra.dat'
			lightcurve_url = 'https://hesma.h-its.org/lib/exe/fetch.php?media=data:models:' + model_name + '_lightcurve.dat'
			lightcurves_early_url = 'https://hesma.h-its.org/lib/exe/fetch.php?media=data:models:' + model_name + '_lightcurves_early.dat'
			density_url = 'https://hesma.h-its.org/lib/exe/fetch.php?media=data:models:' + model_name + '_density.dat'
	
			#Retrieveing the datafiles
			r_isotopes = s.get(isotopes_url, allow_redirects = True)
			r_spectra = s.get(spectra_url, allow_redirects = True)
			r_lightcurve = s.get(lightcurve_url, allow_redirects = True)
			r_lightcurves_early = s.get(lightcurves_early_url, allow_redirects = True)
			r_density = s.get(density_url, allow_redirects = True)

			#Writing the datafiles to disk if they were found
			if r_isotopes.text != 'Not Found' and len(isotope_check) == 0:
				open(root_path + '/isotopes/' + model_name + '_isotopes.dat', 'wb').write(r_isotopes.content)
			if r_spectra.text != 'Not Found' and len(spectra_check) == 0:
				open(root_path + '/spectra/' + model_name + '_spectra.dat', 'wb').write(r_spectra.content)
			if r_lightcurve.text != 'Not Found' and len(lightcurve_check) == 0:
				open(root_path + '/lightcurve/' + model_name + '_lightcurve.dat', 'wb').write(r_lightcurve.content)
			if r_lightcurves_early.text != 'Not Found' and len(lightcurves_early_check) == 0:
				open(root_path + '/lightcurves_early/' + model_name + '_lightcurves_early.dat', 'wb').write(r_lightcurves_early.content)
			if r_density.text != 'Not Found' and len(density_check) == 0:
				open(root_path + '/density/' + model_name + '_density.dat', 'wb').write(r_density.content)

def extract_line(array, string):
	output = ''

	for i in array:
		if string in i:
			output = i 
			break

	return output

def remove_empty_entries(array, floats = False):
	output = []
	for i in array:
		if len(i) > 0:
			if floats:
				output.append(float(i))
			else:
				if i != '\n':
					output.append(i)
	return output

def extract_element(string):
	element = string[0]
	if len(string) > 1:
		try:
			float(string[1])
		except:
			element += string[1]

	return element

class spectrum():
	def __init__(self):
		#self.wl = []
		self.flux = []

class species():
	def __init__(self):
		self.profile = []

class lc():
	def __init__(self):
		self.lum = []
		self.AB = []

class UBVR():
	def __init__(self):
		self.U = lc()
		self.B = lc()
		self.V = lc()
		self.R = lc()

class spectra():
	def __init__(self):
		self.epochs = []
		self.spectra = {}
		self.wl = []

class isotopes():
	def __init__(self):
		self.isotopes = []
		self.elements = []
		self.velocity = []
		self.density = []
	
		self.isotopes_data = {}
		self.elements_data = {}

class lightcurve():
	def __init__(self):
		self.lc = lc()
		self.epochs = []

class lightcurves_early():
	def __init__(self):
		self.UBVR = UBVR()
		self.epochs = []

class density():
	def __init__(self):
		self.velocity = []
		self.density = []

#█░█ █▀▀ █▀ █▀▄▀█ ▄▀█   █▀▀ █░░ ▄▀█ █▀ █▀
#█▀█ ██▄ ▄█ █░▀░█ █▀█   █▄▄ █▄▄ █▀█ ▄█ ▄█

class HESMA_model():
	def __init__(self, spectra_file = None, isotopes_file = None, lightcurve_file = None, lightcurves_early_file = None, density_file = None, model_name = None):
		self.model_name = model_name
		if model_name != None:
			self.time = self.get_time()
			print('\n\n')
			print(text.CYAN + text.BOLD + model_name + text.END + text.BOLD + ':' + text.END)

		if spectra_file != None:
			try:
				self.spectra_read(spectra_file)
			except:
				print('Spectra file not found.')
		if isotopes_file != None:
			try:
				self.isotopes_read(isotopes_file)
			except:
				print('Isotopes file not found.')
		if lightcurve_file != None:
			try:
				self.lightcurve_read(lightcurve_file)
			except:
				print('Lightcurve file not found.')
		if lightcurves_early_file != None:
			try:
				self.lightcurves_early_read(lightcurves_early_file)
			except:
				print('Lightcurves_early file not found.')
		if density_file != None:
			try:
				self.density_read(density_file)
			except:
				print('Density file not found.')

		print('\n')

	def get_time(self):
		login_url = 'https://hesma.h-its.org/doku.php?id=start&do=login&sectok='
		model_url = 'https://hesma.h-its.org/doku.php?id=data:models:' + self.model_name + '_overview'
		with requests.Session() as s:
			p = s.post(login_url, data = payload)
			r_model = s.get(model_url, allow_redirects = True)
			#Retrieving the time at the end of the simulation from the website
			time_line = extract_line(html.fromstring(r_model.text).xpath("//p/text()"), 'Time at the end of this simulation:')
			for i in time_line.split(sep = ' '):
				try:
					time = float(i)
					print(i)
					break
				except:
					continue
		
			return time

	#█▀█ █▀▀ ▄▀█ █▀▄
	#█▀▄ ██▄ █▀█ █▄▀

	def spectra_read(self, spectra_file):
		self.spectra = spectra()
		#Reading in the lines from the .dat file
		with open(spectra_file, 'r') as file:
			lines = file.readlines()
			self.spectra.epochs = remove_empty_entries(lines[0].split(sep = ' '), floats = True)[1:]

		#Constructing and populating the dictionary to hold the spectra
		for i in self.spectra.epochs:
			self.spectra.spectra[i] = spectrum()
			
		#Filling the wl and flux data for each spectrum
		for k in lines[1:]:
			line = remove_empty_entries(k.split(sep = ' '), floats = True)

			self.spectra.wl.append(line[0])
			#for epoch in self.spectra_epochs:
			#	self.spectra[epoch].wl.append(line[0])

			for i in range(1, len(line)):
				self.spectra.spectra[self.spectra.epochs[i-1]].flux.append(line[i])

	def isotopes_read(self, isotopes_file):
		self.isotopes = isotopes()
		#Reading in the lines from the .dat file
		with open(isotopes_file, 'r') as file:
			lines = file.readlines()
			self.isotopes.isotopes = remove_empty_entries(lines[0].split(sep = ' '))[4:]

		#Constructing an element array so that all isotopes of an element can be plotted together
		for iso in self.isotopes.isotopes:
			element = extract_element(iso)
			if element not in self.isotopes.elements:
				self.isotopes.elements.append(element)

		#Constructing and populating the dictionaries to hold the profiles
		for elem in self.isotopes.elements:
			self.isotopes.elements_data[elem] = species()
		for iso in self.isotopes.isotopes:
			self.isotopes.isotopes_data[iso] = species()

		#Filling in the profile data for each of the isotopes and summing the isotopes for the elements
		for k in range(len(lines[1:])):
			line = remove_empty_entries(lines[k+1].split(sep = ' '), floats = True)
			self.isotopes.velocity.append(line[0])
			self.isotopes.density.append(line[1])
			#Creating a new entry in the element profiles to hold the sums of the individual isotope contriubtions	
			for i in range(len(self.isotopes.elements)):
				self.isotopes.elements_data[self.isotopes.elements[i]].profile.append(0)
			#Filling the profile entries
			for h in range(len(self.isotopes.isotopes)):
				self.isotopes.isotopes_data[self.isotopes.isotopes[h]].profile.append(line[h+2])
				self.isotopes.elements_data[extract_element(self.isotopes.isotopes[h])].profile[k] += line[h+2]

	def lightcurve_read(self, lightcurve_file):
		self.lightcurve = lightcurve()
		with open(lightcurve_file, 'r') as file:
			for i in file.readlines():
				line = remove_empty_entries(i.split(sep = ' '), floats = True)
				self.lightcurve.epochs.append(line[0])
				self.lightcurve.lc.lum.append(line[1])

	def lightcurves_early_read(self, lightcurves_early_file):
		self.lightcurves_early = lightcurves_early()

		with open(lightcurves_early_file, 'r') as file:
			for i in file.readlines()[1:]:
				line = remove_empty_entries(i.split(sep = ' '), floats = True)
				self.lightcurves_early.epochs.append(line[0])
				self.lightcurves_early.UBVR.U.AB.append(line[1])
				self.lightcurves_early.UBVR.B.AB.append(line[2])
				self.lightcurves_early.UBVR.V.AB.append(line[3])
				self.lightcurves_early.UBVR.R.AB.append(line[4])

	def density_read(self, density_file):
		self.density = density()
		with open(density_file, 'r') as file:
			for j in file.readlines()[1:]:
				vel0, den0 = remove_empty_entries(j.split(sep = ' '), floats = True)
				self.density.velocity.append(vel0)
				self.density.density.append(den0)

		try:
			check = self.time

			self.density.radius0 = []
			for vel in self.density.velocity:
				self.density.radius0.append(self.time * vel)

		except:
			return

	#█▀█ █░░ █▀█ ▀█▀
	#█▀▀ █▄▄ █▄█ ░█░

	def spectra_plot(self, num_spec = None):
		try:
			#File check - this should fail if there is no imported isotopes file and trigger the exception
			check = self.spectra.epochs[0]	
			#Deciding how many spectra to plot in each figure
			"""
			if num_spec == None:
				num_spec = 10
				cf_checks = [7, 6, 5, 4]
				for l in cf_checks:
					if len(self.spectra_epochs)%l == 0:
						num_spec = l
						break
			"""
			if num_spec == None:
				num_spec = 13
	
			colours = rainbow_array(num_spec)	
			fig = plt.figure(figsize = (12, 10))
			#Plotting the spectra
			for i in range(len(self.spectra.epochs)):
				#plt.plot(self.spectra[self.spectra_epochs[i]].wl, np.array(self.spectra[self.spectra_epochs[i]].flux) + i/10, label = str(self.spectra_epochs[i]) + ' days')
				plt.plot(self.spectra.wl, np.array(self.spectra.spectra[self.spectra.epochs[i]].flux) + i/10, label = str(self.spectra.epochs[i]) + ' days', color = colours[i%num_spec], linewidth = 2)	
				#Setting plot properties and displaying each of the figures
				if (i+1)%num_spec == 0:
					plt.xlabel(r'Wavelength ($\AA$)')
					plt.ylabel('Flux + offset')
					plt.legend()
					if self.model_name != None:
						plt.title(self.model_name + ' Spectra')
					else:
						plt.title('Spectra')
					plt.tight_layout()
					plt.show()
					if i != len(self.spectra.epochs)-1:
						fig = plt.figure(figsize = (12, 10))	
			#Backup formatting and display call in case the number of epochs doesn't divide by 4, 5, 6, 7, or 10
			if (i+1)%num_spec != 0:
				plt.xlabel(r'Wavelength ($\AA$)')
				plt.ylabel('Flux + offset')
				plt.legend()
				if self.model_name != None:
					plt.title(self.model_name + ' Spectra')
				else:
					plt.title('Spectra')
				plt.tight_layout()
				plt.show()

		except:
			#Exception should only display when there has not been a spectrum file imported to the instance of the class
			print(text.RED + 'No associated spectra.dat file' + text.END)

	def elements_plot(self, elements = ['c', 'o', 'mg', 'si', 's', 'ca', 'fe', 'ni']):
		try:
			#File check - this should fail if there is no imported isotopes file and trigger the exception
			check = self.isotopes.velocity[0]

			#Establishing the colours array and creating the figure
			colours = rainbow_array(len(elements))
			fig = plt.figure(figsize = (12, 6))
			ax = fig.add_subplot(1, 1, 1)
	
			#Plotting each of the element profiles
			for l in range(len(elements)):
				ax.plot(self.isotopes.velocity, self.isotopes.elements_data[elements[l]].profile, label = elements[l], color = colours[l], linewidth = 2)
	
			#Setting figure parameters
			plt.legend()
			plt.xlabel(r'Velocity ($km$ $s^{-1}$)')
			plt.ylabel('Mass fraction')
			plt.ylim(0, 1)
			if self.model_name != None:
				plt.title(self.model_name + ' Elemental Abundances')
			else:
				plt.title('Elemental Abundances')
			plt.gca().set_xlim(left=0)
			plt.tight_layout()
			plt.show()
		except:
			#Exception should only display when there has not been a spectrum file imported to the instance of the class
			print(text.RED + 'No associated isotopes.dat file' + text.END)

	def isotopes_plot(self, isotopes = ['c12', 'o16', 'si28', 'ni56'], add_isotopes = None):
		try:
			#File check - this should fail if there is no imported isotopes file and trigger the exception
			check = self.isotopes.velocity[0]

		except:
			#Exception should only display when there has not been a isotopes file imported to the instance of the class
			print(text.RED + 'No associated isotopes.dat file' + text.END)
			return

		try:
			#Adding any isotopes to the 4 default
			if add_isotopes != None:
				for i in add_isotopes:
					try:
						isotopes.append(i)
					except:
						print(str(i) + ' is not valid')

			#Establishing the colours array and creating the figure
			colours = rainbow_array(len(isotopes))
			fig = plt.figure(figsize = (12, 6))
			ax = fig.add_subplot(1, 1, 1)
	
			#Plotting each of the element profiles
			for l in range(len(isotopes)):
				ax.plot(self.isotopes.velocity, self.isotopes.isotopes_data[isotopes[l]].profile, label = isotopes[l], color = colours[l], linewidth = 2)
	
			#Setting figure parameters
			plt.legend()
			plt.xlabel(r'Velocity ($km$ $s^{-1}$)')
			plt.ylabel('Mass fraction')
			plt.ylim(0, 1)
			if self.model_name != None:
				plt.title(self.model_name + ' Isotopic Abundances')
			else:
				plt.title('Isotopic Abundances')
			plt.gca().set_xlim(left=0)
			plt.tight_layout()
			plt.show()
		except:
			#Exception should only display when there has not been a isotopes file imported to the instance of the class
			print('Invalid isotope')

	def lightcurve_plot(self):
		try:
			#File check - this should fail if there is no imported lightcurve file and trigger the exception
			check = self.lightcurve.epochs[0]

			fig = plt.figure(figsize = (12, 6))
			ax = fig.add_subplot(1, 1, 1)
			ax.plot(self.lightcurve.epochs, self.lightcurve.lc.lum, color = '#C70039', linewidth = 2)
	
			plt.xlabel('Days relative to explosion')
			plt.ylabel(r'Luminosity ($erg$ $s^{-1}$)')
			if self.model_name != None:
				plt.title(self.model_name + ' UVOIR Lightcurve')
			else:
				plt.title('UVOIR Lightcurve')
			plt.tight_layout()
			plt.show()

		except:
			#Exception should only display when there has not been a lightcurve file imported to the instance of the class
			print(text.RED + 'No associated lightcurve.dat file' + text.END)
			return

	def lightcurves_early_plot(self):
		try:
			check = self.lightcurves_early.epochs[0]

			fig = plt.figure(figsize = (12, 6))
			ax = fig.add_subplot(1, 1, 1)
	
			ax.plot(self.lightcurves_early.epochs, self.lightcurves_early.UBVR.U.AB, color = U_colour, label = 'U')
			ax.plot(self.lightcurves_early.epochs, self.lightcurves_early.UBVR.B.AB, color = B_colour, label = 'B')
			ax.plot(self.lightcurves_early.epochs, self.lightcurves_early.UBVR.V.AB, color = V_colour, label = 'V')
			ax.plot(self.lightcurves_early.epochs, self.lightcurves_early.UBVR.R.AB, color = R_colour, label = 'R')
	
			plt.gca().invert_yaxis()
			plt.xlabel('Days since explosion')
			plt.ylabel('Absolute AB magnitude')
			if self.model_name != None:
				plt.title(self.model_name + ' Early Lightcurves')
			else:
				plt.title('Early Lightcurves')
			plt.tight_layout()
			plt.legend()
			plt.show()
		except:
			#Exception should only display when there has not been an early lightcurves file imported to the instance of the class
			print(text.RED + 'No associated lightcurves_early.dat file' + text.END)
			return

	def density_plot(self):
		try:
			check = self.density.velocity[0]

			fig = plt.figure(figsize = (12, 6))
			ax = fig.add_subplot(1, 1, 1)

			ax.plot(self.density.velocity, self.density.density)

			plt.xlabel(r'Velocity ($km$ $s^{-1}$)')
			plt.ylabel(r'Density ($g$ $cm^{-3}$)')
			if self.model_name != None:
				plt.title(self.model_name + ' Density')
			else:
				plt.title('Density')
			plt.tight_layout()
			plt.show()

		except:
			#Exception should only display when there has not been a density file imported to the instance of the class
			print(text.RED + 'No associated density.dat file' + text.END)
			return

	def density_plot_epoch(self, day):
		try:
			check = self.density.velocity[0]
			check = self.density.radius0[0]

			t = day * 24*3600
			sf = []
			for i in range(len(self.density.radius0)):
				sf.append( (self.density.radius0[i])**3/(self.density.radius0[i] + t*self.density.velocity[i])**3 )

			new_density = []
			for i in range(len(self.density.density)):
				new_density.append(self.density.density[i] * sf[i])

			plt.plot(self.density.velocity, new_density)
			plt.show()

		except:
			#Exception should only display when there has not been a density file imported to the instance of the class or the assingment of radius0 has gone wrong
			print(text.RED + 'No associated density.dat file or there has been an issue callibrating the inital radii' + text.END)
			return

def load_HESMA_model(model_name, local = False):
	if not local:
		download_HESMA_data(model_name)
	return HESMA_model(model_name = model_name,\
		spectra_file = root_path + '/spectra/' + model_name + '_spectra.dat',\
		isotopes_file = root_path + '/isotopes/' + model_name + '_isotopes.dat',\
		lightcurve_file = root_path + '/lightcurve/' + model_name + '_lightcurve.dat',\
		lightcurves_early_file = root_path + '/lightcurves_early/' + model_name + '_lightcurves_early.dat',\
		density_file = root_path + '/density/' + model_name + '_density.dat')
