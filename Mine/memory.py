import variables_CasaCarles as var
from functions import *

starttime = time.time()

while (var.cap.isOpened):

    ret, frame = var.cap.read()

    if ret == True:
        start_algorithm_time = time.time()
        df = pd.DataFrame()
        bbinfo = pd.DataFrame(columns=['Frame', 'ID', 'maxID', 'date', 'time', 'topleft', 'width', 'height', 'center', 'velocity'])

        var.q.append(frame)
        if len(var.q) == int(var.ufps):
            start_creating_contours = time.time()
            medianFrame, contours = creating_countours(var.q)
            end_creating_contours = time.time()
            var.execution_time_file.write('Creating contours: {}\n'.format(end_creating_contours-start_creating_contours))
            realFrame = cv2.resize(var.q[var.ufps-1], var.dim)
            '''imgplot = plt.imshow(realFrame)
            plt.show()'''
            start_contour_calculations = time.time()
            var.ident, curr, centers, bbinfo, count_people, fg_bounding = contour_calculations(contours,
                                                                                               var.ident,
                                                                                               medianFrame,
                                                                                               realFrame,
                                                                                               bbinfo,
                                                                                               var.Icount,
                                                                                               var.memory)
            end_contour_calculations = time.time()
            var.execution_time_file.write('Contour calculations: {}\n'.format(end_contour_calculations-start_contour_calculations))

            in_counting(var.zones, curr)
            out_counting(var.zones, curr)


            shop_space = creation_in_out_spaces(var.veritas)
            pts = np.array(shop_space, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(realFrame, [pts], True, (255, 0, 0), thickness = 2)

            shop_space = creation_in_out_spaces(var.zona1)
            pts = np.array(shop_space, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(realFrame, [pts], True, (255, 0, 0), thickness = 2)

            shop_space = creation_in_out_spaces(var.zona2)
            pts = np.array(shop_space, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(realFrame, [pts], True, (255, 0, 0), thickness = 2)

            start_part1 = time.time()
            df = df.append({'Frame': var.Icount, 'Count': count_people}, ignore_index=True)
            info = [
                ("In", var.total_in[2]),
                ("Out", var.total_out[2]),
                ("Now counted", count_people),
            ]

            info_derecha = [
                ("In", var.total_out[1]),
                ("Out", var.total_in[1]),
            ]

            info_izquierda = [
                ("In", var.total_out[0]),
                ("Out", var.total_in[0]),
            ]

            cv2.putText(realFrame, '{}'.format(var.Icount), (0, 15), 0, 0.5, (255, 255, 255), 2)


            # loop over the info tuples and draw them on our frame
            cv2.rectangle(realFrame, (0, var.dim[1]-(((len(info)+1)*20))), (len('Total counted')*15, var.dim[1]), (255,255,255), thickness=-1)
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(realFrame, text, (10, var.dim[1] - ((i * 20) + 20)),
                            16, 0.5, (0, 0, 0), 1)

            cv2.rectangle(realFrame, (var.W-len('counted')*15, 200), (var.W, 200+(((len(info_derecha)+1)*20))), (255,255,255), thickness=-1)
            for (i, (k, v)) in enumerate(info_derecha):
                text = "{}: {}".format(k, v)
                cv2.putText(realFrame, text, (var.W-len('counted')*15+10, (200+(((len(info_derecha)+1)*20))) - ((i * 20) + 20)),
                            16, 0.5, (0, 0, 0), 1)

            cv2.rectangle(realFrame, (0, 200),
                          (len('counted') * 15, 200 + (((len(info_izquierda) + 1) * 20))), (255, 255, 255), thickness=-1)
            for (i, (k, v)) in enumerate(info_izquierda):
                text = "{}: {}".format(k, v)
                cv2.putText(realFrame, text, (10, (200 + (((len(info_izquierda) + 1) * 20))) - ((i * 20) + 20)),
                            16, 0.5, (0, 0, 0), 1)
            end_part1 = time.time()
            var.execution_time_file.write('Part1: {}\n'.format(end_part1 - start_part1))

            #preparat per fer el grafic del temps mitjà en una tenda
            '''var.x.append(var.Icount)
            var.y.append(psutil.cpu_percent())

            var.ax.plot(var.x, var.y, color='b')
            var.fig.canvas.draw()
            var.ax.set_xlim(left=max(0, var.Icount - 50), right=var.Icount + 50)'''

            start_part2 = time.time()
            temp = pd.merge(df, bbinfo, on='Frame')
            var.new = var.new.append(temp)
            end_part2_1 = time.time()
            var.execution_time_file.write('Part 2.1: {}\n'.format(end_part2_1-start_part2))
            #OPTIMIZAR ESTO
            #var.new.to_excel('DataOutputs/new.xlsx', index = False)
            start_part2_2 = time.time()
            cv2.imshow('frame', realFrame)
            var.out.write(realFrame)
            #cv2.imwrite('{}/bounding{}.jpg'.format(var.directory, var.Icount), fg_bounding)
            #cv2.imwrite('{}/frame{}.jpg'.format(var.directory, var.Icount), realFrame)
            end_part2_2 = time.time()
            var.execution_time_file.write('Part 2.2: {}\n'.format(end_part2_2-start_part2_2))
            start_part2_3 = time.time()
            var.Icount += 1
            var.q.clear()
            end_part2 = time.time()
            var.execution_time_file.write('Part 2.3: {}\n'.format(end_part2-start_part2_3))
            var.execution_time_file.write('Part2: {}\n'.format(end_part2-start_part2))
            start_part3 = time.time()
            if var.Icount > 1:
                if len(var.memory) == var.memory_frames:
                    # if var.Icount != (((Nframes - Nframes%var.ufps)/var.ufps)-1):
                    var.memory.append(curr)
                    var.memory.popleft()
                else:
                    var.memory.append(curr)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

            final_algorithm_time = time.time()
            var.execution_time_file.write('Part3: {}\n'.format(final_algorithm_time-start_part3))
            var.execution_time_file.write('Algorithm: {}\n'.format(final_algorithm_time-start_algorithm_time))


    else:
        break
#display(var.new)
memory = var.memory
var.new.to_excel('DataOutputs/new.xlsx', index = False)
print('DATA SAVED!')
var.cap.release()
var.out.release()
cv2.destroyAllWindows()
