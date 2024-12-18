import pykep as pk
    
# Use the JPL low precision ephemerides
from pykep.planet import jpl_lp 
    
from matplotlib import pyplot as plt
import numpy as np



def main():

    print('\nInteplanetary porkchop plot using PyKEP\n')
    #Pykep makes use of lambert transfers to study interplanetary trajectories by producing Pork-Chop plots
    
    # Define the start and end planets of the inteplanetary trajectory

    while True:
        start_planet_input = input('Enter initial planet: ').lower()   
        end_planet_input = input('Enter destination planet: ').lower() 
    
        if(start_planet_input == end_planet_input):
            print('Incorrect input. Initial and destination planets cannot be the same.\n')
        elif(start_planet_input or end_planet_input) not in ('mercury','venus','earth','mars','jupiter','saturn','uranus','neptune','pluto'):
            print('Unknown planet name.\n')
        else:
            break
        
    start_planet = jpl_lp(start_planet_input)
    end_planet = jpl_lp(end_planet_input)
    
    # Set departure epochs and transfer time
    
    while True:
        try:
            input_eph1 = float(input('\nEnter first departure epoch in MJD2000: '))
            input_eph2 = float(input('Enter latest departure epoch in MJD2000: '))
        except ValueError:
            print('Invalid input.\n')
            continue
        else:
            print(f'\nDepature window is from {input_eph1} to {input_eph2} (MJD2000)\n')
            break
    
    while True:
        try:
            input_flt1 = float(input('Enter minimum flight time in days: '))
            input_flt2 = float(input('Enter maximum flight time in days: '))
        except ValueError:
            print('Invalid input.\n')
            continue
        else:
            print(f'\nTotal mission duration is from {input_flt1} to {input_flt2} (days)')
            break
    
    # Sample departure epochs and transfer times every 15 days and solve the lambert problem in a large defined grid
    
    plotContours(start_planet,end_planet,input_eph1,input_eph2,input_flt1,input_flt2,15.0)
    
    while True:
        zoom = input('\nWould you like to sample with a finer resolution? (Y/N)\n')
        if zoom[0].upper() == 'Y':
            pass
            while True:
                try:
                    input_new_eph1 = float(input('\nEnter the updated first departure epoch in MJD2000: '))
                    input_new_eph2 = float(input('Enter the updated latest departure epoch in MJD2000: '))
                except ValueError:
                    print('Invalid input.\n')
                    continue
                else:
                    print(f'\nDepature window is from {input_new_eph1} to {input_new_eph2} (MJD2000)\n')
                    break
            
            while True:
                try:
                    input_new_flt1 = float(input('Enter the updated minimum flight time in days: '))
                    input_new_flt2 = float(input('Enter updated maximum flight time in days: '))
                except ValueError:
                    print('Invalid input.\n')
                    continue
                else:
                    print(f'\nTotal mission duration is from {input_new_flt1} to {input_new_flt2} (days)')
                    break
            
            plotContours(start_planet,end_planet,input_new_eph1,input_new_eph2,input_new_flt1,input_new_flt2,1.0)
            
        else: 
            print('Program ended.\n')
            break
    
    
def plotContours(start_planet,end_planet,eph1,eph2,flt1,flt2,sampling_size):

    start_epochs = np.arange(eph1,eph2,sampling_size)
    duration = np.arange(flt1,flt2,sampling_size)
    data=list()
    for start in start_epochs:
        row = list()
        for T in duration:
            r1,v1 = start_planet.eph(pk.epoch(start))
            r2,v2 = end_planet.eph(pk.epoch(start+T))
            l = pk.lambert_problem(r1,r2,T*60*60*24, start_planet.mu_central_body)
            DV1 = np.linalg.norm(np.array(v1)-np.array(l.get_v1()[0]))
            DV2 = np.linalg.norm(np.array(v2)-np.array(l.get_v2()[0]))
            DV1 = max([0,DV1-eph1])
            DV = DV1+DV2
            row.append(DV)
        data.append(row)

    minrows = [min(l) for l in data]
    i_idx = np.argmin(minrows)
    j_idx = np.argmin(data[i_idx])
    best = data[i_idx][j_idx]
    # 示例：将MJD2000值9658转换为日期
    year, month, day = mjd2000_to_date_manual(start_epochs[i_idx])
    
    print('\nBest DV: ',best)
    print('Launch epoch (MJD2000): ',start_epochs[i_idx])
    print(f"{year}-{month}-{day}")
    print('Duration (days): ',duration[j_idx])
    duration_pl2, start_epochs_pl2 = np.meshgrid(duration, start_epochs)

    CP2 = plt.contourf(start_epochs_pl2,duration_pl2,np.array(data),levels=list(np.linspace(best,5000,10)))
    plt.colorbar(CP2).set_label('△V km/s')  
    plt.title(f'{start_planet.name} - {end_planet.name} Total △V Requirements'.title())
    plt.xlabel('Launch Date (MJD2000)')
    plt.ylabel('Mission Duration (days)')
    plt.show()
    
def mjd2000_to_date_manual(mjd2000_value):
    # MJD2000的基准日期是2000年1月1日，其对应的MJD2000值为0
    # 我们需要计算从2000年1月1日起，经过mjd2000_value天后的日期

    # 初始化基准日期的年、月、日
    year = 2000
    month = 1
    day = 1

    # 计算总天数
    total_days = mjd2000_value

    # 月份的天数，不考虑闰年
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # 循环，直到总天数减到0
    while total_days > 0:
        # 如果是闰年，2月有29天
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            month_days[1] = 29
        else:
            month_days[1] = 28

        # 如果剩余天数大于当前月的天数，减去这个月的天数，并进入下一个月
        if total_days >= month_days[month - 1]:
            total_days -= month_days[month - 1]
            month += 1
            # 如果月份超过12，进入下一年
            if month > 12:
                month = 1
                year += 1
        else:
            # 如果剩余天数小于当前月的天数，这就是我们要找的日期
            day += total_days
            total_days = 0

    return year, month, day




if __name__ == "__main__":
	main()
