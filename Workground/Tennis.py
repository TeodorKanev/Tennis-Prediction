#!/usr/bin/env python
# coding: utf-8

# # Тенис анализ

# ## Изкувствен интелект 2020/2021

# Основната задача на този материал е да се опитаме да предскажем резултат от даден тенис мач, на базата на изминали такива. Преди да започнем да решаваме проблема ще представим и изясним данните, които ползваме, както и да направим обобщен анализ на цялостната игра - важни качества на играч за по-вероянтна победа, зависимост между корта и статистики за мача, и още. След като създадем определен модел и оценим неговата прецизност ще видим как този модел се справя в света на залозите. Нека видим дали благодарение на него, можем да се пенсионираме още днес.

# ## 1. За данните
# 
# Преди всичко, могат да бъдат изтеглени от https://github.com/JeffSackmann/tennis_atp, като ресурса е изцяло на собственика. Доста различни версии на този набор от данни могат да бъдат намерени, но тук са използвани от линка по-горе с последна промяна - 23 Януари 2021.
# 
# Те събират в себе си сингъл мачове на мъже от най-елитната асоциация за тенис по света - ATP в периода 1968 - 2021 година. Доста информация има за даден мач, като някой от характеристиките са:
# 
# 
# 
# -['tourney_name'] - име на турнира.
# 
# -['surface']- настилка на корта.
# 
# -['winner/loser_ht']- височина на победителя/загубилия.
# 
# -['winner/loser rank']- ранкинг на победителя/загубилия.
# 
# -['w/l_ace']- брой асове на побеителя/загубилия.
# 
# -['w/l_1stIn']- процент вкаран първи сервиз.
# 
# -['winner/loser_name']- име на играчите.
# 

# ## 2. Анализ на зависимостите в тениса

# Първо ще заредим данните и нужните библиотеки

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('data/matches/all_matches.csv', low_memory=False)


# Това са всички характеристики:

# In[3]:


print(df.columns)


# Сега ще разгледаме някой интересни факти за тениса като например кой има най-много победи. Мислите ли,че това е Федерер или Надал?

# In[4]:


plt.subplots(figsize=(18,6))
sns.barplot(x = df['winner_name'].value_counts()[:10].index,y = df['winner_name'].value_counts()[:10].values,palette='hot')
plt.title('Top players by matches won')
plt.show()


# И аз не бях чувал Джими Конърс, но както във всеки спорт в миналото е имало спортист, който е бил тотален фаворит в по-голямат част от времето. Ще видим дали Федерер ще го задмине.

# А това е хистограма за най-много асове на победителя в един мач

# In[5]:


plt.subplots(figsize=(18,6))
sns.barplot(x = (df.nlargest(10,['w_ace']))['winner_name'].values,y = (df.nlargest(10,['w_ace']))['w_ace'].values,palette='twilight')
plt.title('Most aces in a winning match')
plt.show()


# Нека да видим същите при губещ мач.

# In[6]:


plt.subplots(figsize=(18,6))
sns.barplot(x = (df.nlargest(10,['l_ace']))['loser_name'].values,y = (df.nlargest(10,['l_ace']))['l_ace'].values,palette='twilight')
plt.title('Most aces in a losing match')
plt.show()


# Интересно как се губи мач с над 100 аса. Ами като опонента ти е направил повече от това дори. Точно така - и двамата постигат тези рекорди в един и същ мач. Мача на Джон Иснър и Никола Маю през 2010 година на Уимбълдън влезе в историята на тениса като най-дългия тенис мач - 11 часа и 5 минути при игра от три дни. Правилата тогава бяха такива, че петият сет се играе по правилата на тайбрек - първи до 7 гейма и разлика 2. Сета завършва 70-68 гейма в полза на Джон. 

# Дали правенето на асове си е изцяло умение?

# In[7]:


plt.subplots(figsize=(18,6))
sns.barplot(x = (df.nlargest(10,['w_ace']))['winner_name'].values,y = (df.nlargest(10,['w_ace']))['winner_ht'].values,palette='twilight')
plt.title('Height of players with most aces')
plt.show()


# Ами оказва се, че не съвсем. Челните места в класацията се държат от доста високи тенисисти. Това е логично, тъй като височината ти позволява по-остра траектория на първия удар.

# Това са най-младите победители.

# In[8]:


plt.subplots(figsize=(18,6))
sns.barplot(x = (df.nsmallest(5,['winner_age']))['winner_name'].values,y = (df.nsmallest(5,['winner_age']))['winner_age'].values,palette='rocket')
plt.title('The youngest winners')
plt.show()


# Нека видим има ли значение настилката за това колко аса ще има в мача

# In[9]:


surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
values = [(df[df['surface'] == i]['w_ace'].sum() + df[df['surface'] == i]['l_ace'].sum()) for i in surfaces]


# In[10]:


plt.subplots(figsize=(18,6))
sns.barplot(x = surfaces,y = values,palette=['darkblue', 'brown','darkgreen','orange'])
plt.title('Aces on different surface')
plt.show()


# Това са асовете от всички мачове на дадената настилка, но твърдата настилка е най-евтина за подръжка и използвана във всички условя, докато при тревата нещата са по съвсем друг начин. Нека да видим средно по колко аса има на дадена настилка

# In[11]:


values = [(df[df['surface'] == i]['w_ace'].sum() + df[df['surface'] == i]['l_ace'].sum()) / len(df[df['surface'] == i]) for i in surfaces]


# In[12]:


plt.subplots(figsize=(18,6))
sns.barplot(x = surfaces,y = values,palette=['darkblue', 'brown','darkgreen','orange'])
plt.title('Average aces per match on different surface')
plt.show()


# Така данните са по-логични, тъй като тревата е бърза настилка и предлага условия за създаване на асове. Клея от друга страна е доста бавна и там имаме най-малко. Тук е важно да отбележим, че има 2 турнира от голям шлем на твърда настилка и само 1 на трева, така че 5 сетовите мачове са повече на твърда настилка,а от там и възможността за асове.

# Нека разгледаме разликите при стойностите характеризиращи победител и загубил.

# In[13]:


compareFeatures = ['winner_rank','loser_rank']

plt.figure(1, figsize=(20,12))

for i in range(0,len(compareFeatures)):
    plt.subplot(1,2,i+1)
    df[compareFeatures[i]].plot.hist(title=compareFeatures[i], bins = 20, xticks = [0,100,200,300,400,500,100,1500,2000])


# Тук има някакви дребни разминавания - играчите с по-висок ранк изглежда да печелят по-често, но не можем да заклчючим това само с тази диаграмач тъй като и в доста от мачовете имаме двама играчи с ранк под 100.

# За това да видим в колко процента от мачовете играча с по-висок ранк печели.

# In[14]:


"{0:.0f}%".format(len(df.query('loser_rank > winner_rank'))/len(df)*100)


# Това изглежда прекалено нисък процент. А обратното?

# In[15]:


"{0:.0f}%".format(len(df.query('winner_rank > loser_rank'))/len(df)*100)


# Тъй като не може двама различни играча да са с еднакъв ранк, заключваме, че има мачове в които нямаме информация за това. Сега ще проверим

# In[16]:


df['winner_rank'].isnull().sum() + df['loser_rank'].isnull().sum()


# Тъй като може да има мачове в които нямаме информация и за двамата играчи, поне в 42 000 от случайте нямаме информация. Нека вземем процентно на това което имаме.

# In[17]:


print("{0:.0f}%".format((49/74)*100))
print("{0:.0f}%".format((25/74)*100))


# Това вече изглежда по-правдоподобно. И като отчетем, че поради начина на пресмятане на ранк в АТП, ако играч се контузи и не участва в турнири, той губи позиции - това може би е коректно разпределение

# А дали височината помага за крайния резултат?

# In[18]:


compareFeatures = ['winner_ht','loser_ht']

plt.figure(1, figsize=(20,12))

for i in range(0,len(compareFeatures)):
    plt.subplot(1,2,i+1)
    df[compareFeatures[i]].plot.hist(title=compareFeatures[i], xticks = [160,165,170,175,180,185,190,195,200,205,210])


# По-скоро не е толкова голям фактор.

# На следващата хистограма, ще видим, че и асовете не са.

# In[19]:


compareFeatures = ['w_ace','l_ace']

plt.figure(1, figsize=(20,12))

for i in range(0,len(compareFeatures)):
    plt.subplot(1,2,i+1)
    df[compareFeatures[i]].plot.hist(title=compareFeatures[i], xlim = (0,10),bins = 80, xticks= list(range(0,10,1)))


# И последно ще видите матрица за корелацията между характеристиките.

# In[20]:


correlation_matrix = df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlation_matrix, annot= True, linewidth=0.1, cmap= 'icefire')


# Основно виждаме четири сегмента с висока корелация между характеристиките свързани пряко със самия мач на победителя и загубилия. Това е така, защото тези характеристики донякъде определят целия мач и себе си.
# Процента на вкаран първи сервиз почти не е зависим от ранка на играча.

# ## 3. Обработване на данните

# Първо ще си изберем по-малък интервал във времето. Доста от правилата, технологиите и самата игра е била коренно различна преди 30 години например. Нека разгледаме мачовете от 1 Януари 2000 година до сега. Двадесет години са достатъчно голям интервал, а и мачовете са достатъчно.

# In[21]:


df = df[df['tourney_date'] > 20000101]


# In[22]:


len(df)


# Следва да премахнем характеристиките, които няма да ползваме в модела и естествено тези за които нямаме информация преди самия мач. Това са тези, следящи статистики за самия мач.

# In[23]:


df = df.drop(columns = ['score', 'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon','w_2ndWon','w_SvGms','w_bpSaved','w_bpFaced','l_ace','l_df','l_svpt','l_1stIn','l_1stWon','l_2ndWon','l_SvGms','l_bpSaved','l_bpFaced','winner_name', 'loser_name', 'tourney_id'])


# Това са останалите

# In[24]:


print(df.columns)


# Най-вероятно доста от тях ще имат неизвестни стойности.

# In[25]:


df.isna().sum()


# За да добием по-голяма представа нека ги видим като процент от всичките данни.

# In[26]:


df.isna().sum()/len(df)*100


# Имаме доста високи проценти за някой характеристики, но като цяло не са толкова много като бройка и повечето могат да се оправят. Ще примахнем тези с над 58% липса. Височината, годините и  ранка ще ги запълним със средното за съответната колона, а настилката и ръката с която играе ще им обърнем специално внимание.

# In[27]:


df = df.drop(columns = ['winner_seed', 'loser_seed', 'winner_entry', 'loser_entry'])


# In[28]:


winner_height_mean = df['winner_ht'].sum()/len(df)
loser_height_mean = df['loser_ht'].sum()/len(df)


# In[29]:


df['winner_ht'] = df['winner_ht'].fillna(winner_height_mean)
df['loser_ht'] = df['loser_ht'].fillna(loser_height_mean)


# Да проверим дали сме заместили празните както трябва или има все още празни стойности.

# In[30]:


df['winner_ht'].isna().any() or df['loser_ht'].isna().any()


# In[31]:


winner_rank_mean = df['winner_rank'].sum()/len(df)
winner_rank_points_mean = df['winner_rank_points'].sum()/len(df)
loser_rank_mean = df['loser_rank'].sum()/len(df)
loser_rank_points_mean = df['loser_rank_points'].sum()/len(df)
winner_age_mean = df['winner_age'].sum()/len(df)
loser_age_mean = df['loser_age'].sum()/len(df)


# In[32]:


df['winner_rank'] = df['winner_rank'].fillna(winner_rank_mean)
df['loser_rank'] = df['loser_rank'].fillna(loser_rank_mean)
df['winner_rank_points'] = df['winner_rank_points'].fillna(winner_rank_points_mean)
df['loser_rank_points'] = df['loser_rank_points'].fillna(loser_rank_points_mean)
df['winner_age'] = df['winner_age'].fillna(winner_age_mean)
df['loser_age']= df['loser_age'].fillna(loser_age_mean)


# In[33]:


df.isna().sum()/len(df)*100


# Нека опитаме да вземем информация от други мачове на същия играч за това с каква ръка играе. 

# In[34]:


df[df['winner_hand'].isna()]['tourney_name']


# Липсващите данни са за мачове от купа Дейвис. Не можем да бъдем сигурни кой точно от отбора е играл в този мач без да проверим ръчно всеки. Нека сложим най-вероятната стойност. Очакваме това да е "дясна", но нека проверим.

# In[35]:


df['winner_hand'].value_counts()


# In[36]:


df['loser_hand'].value_counts()


# In[37]:


df['winner_hand'] = df['winner_hand'].fillna('R')
df['loser_hand'] = df['loser_hand'].fillna('R')
df['winner_hand'] = df['winner_hand'].map({'U': 'R', 'L' : 'L', 'R' : 'R'})
df['loser_hand'] = df['loser_hand'].map({'U': 'R', 'L' : 'L', 'R' : 'R'})


# Сега да видим кои са липсващите настилки.

# In[39]:


df[df['surface'].isna()]['tourney_name'].value_counts()


# Отново имаме проблем за този турнир. Мачовете се играят на различни настилки дори в рамките на една група. Най-добре да премахнем тези мачове.

# In[40]:


df.dropna(axis = 0, how='any', inplace=True)


# Вече нямаме липсващи стойности.

# In[41]:


df.isna().any()


# Нека сега видим какви характеристики оставихме, от кои имаме нужда и какви нови можем да направим.

# In[42]:


df.columns


# Държавата на самите играчи не е от значение, тъй като самите тях имаме как да ги идентифицираме. Номера на мача също е ненужен, тъй като знаем в кой кръг се играе. Махаме ги

# In[43]:


df = df.drop(columns = ['winner_ioc', 'loser_ioc'])
df = df.drop(columns=['match_num'])


# Вижте как е представена датата за мача.

# In[44]:


df['tourney_date']


# Повече информация ще получим ако отделим годината и месеца в две отделни характеристики, а деня не е от значение, имайки информация за турнира.

# In[45]:


df['tourney_year'] = df['tourney_date'].astype(str).str[:4].astype(int)
df['tourney_month'] = df['tourney_date'].astype(str).str[4:6].astype(int)
df = df.drop(columns = ['tourney_date'])


# Можем да съставим две нови характеристики - разлика в ранковете и разлика в точките за ранклистата между играчите. Да ние имаме поотделно тези стойности, но често когато от тях създадем нова информация специално за тяхната разлика моделите дават по добри резултати.

# In[46]:


df['rank_difference'] = df['winner_rank'] - df['loser_rank']
df['rank_points_difference'] = df['winner_rank_points'] - df['loser_rank_points']


# Преди да сложим данните в модела трябва да забележим следното: Ние имаме информация за самия победител и загубил, но по време на разпознаване на даден мач, ние не знаем характеристиките на играчите дали спадат в частта при загубил или победител. Следователно не можем да използваме такъв модел за предсказване. Ще трябва по някакъв начин да представим данните вместо победител и загубил като първи и втори играч. Но ако просто преименуваме характеристиките така, ние винаги имаме победител като първи, а загубил като втори играч. Тогава модела ще предсказва винаги първия играч да печели. За да се справим с този проблем ще имаме отново двама играчи, но с равновероятно разпределние ще взимаме число 0 или 1. Това число ще определя дали първия играч ще вземе данните на победителя или на загубилия. Тъй като разпределението е равновероятно и данните са достатъчно големи, можем да предполагаме равномерно разпределени 0 и 1-ци - тоест първия играч ще е победител в близо толкова случаи колкото е и губещ.

# In[47]:


np.random.seed(1234)
first_player_id = []
second_player_id = []
first_player_hand = []
second_player_hand = []
first_player_ht = []
second_player_ht = []
first_player_age =[]
second_player_age = []
first_player_rank = []
second_player_rank = []
first_player_rank_points = []
second_player_rank_points = []
labels = []
for i in range(0,len(df)):
    choice = np.random.choice([0,1],1)
    if (choice == 0):
        first_player_id.append(df.iloc[i]['winner_id'])
        second_player_id.append(df.iloc[i]['loser_id'])
        first_player_hand.append(df.iloc[i]['winner_hand'])
        second_player_hand.append(df.iloc[i]['loser_hand'])
        first_player_ht.append(df.iloc[i]['winner_ht'])
        second_player_ht.append(df.iloc[i]['loser_ht'])
        first_player_age.append(df.iloc[i]['winner_age'])
        second_player_age.append(df.iloc[i]['loser_age'])
        first_player_rank.append(df.iloc[i]['winner_rank'])
        second_player_rank.append(df.iloc[i]['loser_rank'])
        first_player_rank_points.append(df.iloc[i]['winner_rank_points'])
        second_player_rank_points.append(df.iloc[i]['loser_rank_points'])
    else:
        first_player_id.append(df.iloc[i]['loser_id'])
        second_player_id.append(df.iloc[i]['winner_id'])
        first_player_hand.append(df.iloc[i]['loser_hand'])
        second_player_hand.append(df.iloc[i]['winner_hand'])
        first_player_ht.append(df.iloc[i]['loser_ht'])
        second_player_ht.append(df.iloc[i]['winner_ht'])
        first_player_age.append(df.iloc[i]['loser_age'])
        second_player_age.append(df.iloc[i]['winner_age'])
        first_player_rank.append(df.iloc[i]['loser_rank'])
        second_player_rank.append(df.iloc[i]['winner_rank'])
        first_player_rank_points.append(df.iloc[i]['loser_rank_points'])
        second_player_rank_points.append(df.iloc[i]['winner_rank_points'])
    labels.append(choice)


# Променяме характеристиките да са за първи и втори играч и в характеристиката label пазим кой от двамата е победителят.

# In[48]:


df['winner_id'] = first_player_id
df['loser_id'] = second_player_id
df['winner_hand'] = first_player_hand
df['loser_hand'] = second_player_hand
df['winner_ht'] = first_player_ht
df['loser_ht'] = second_player_ht
df['winner_age'] = first_player_age
df['loser_age'] = second_player_age
df['winner_rank'] = first_player_rank
df['loser_rank'] = second_player_rank
df['winner_rank_points'] = first_player_rank_points
df['loser_rank_points'] = second_player_rank_points
df['outcome'] = labels
df['outcome'] = df['outcome'].astype('int')


# In[49]:


df.rename(columns = {'winner_id':'first_player_id', 'winner_hand':'first_player_hand', 'winner_ht':'first_player_ht'
                    ,'winner_age':'first_player_age','winner_rank':'first_player_rank','winner_rank_points':'first_player_rank_points'
                     ,'loser_id':'second_player_id', 'loser_hand':'second_player_hand','loser_ht':'second_player_ht'
                    ,'loser_age':'second_player_age','loser_rank':'second_player_rank','loser_rank_points':'second_player_rank_points'}, inplace = True)


# Тук е важно да променим и новосъздадените характеристики за разлика между ранковете, защото иначе тя ни дава информация за точната разлика между победителя и загубилия, а след като имаме ранковете на първия и втория играч, лесно можем да предскажем кой е победил. Тази информация точно в този вид така или иначе я нямаме при предсказване. Но имаме разликата от ранка на първия и втория така че ще я представим по този начин.

# In[50]:


df['rank_difference'] = df['first_player_rank'] - df['second_player_rank']
df['rank_points_difference'] = df['first_player_rank_points'] - df['second_player_rank_points']


# Така изглеждат в краен вид всички характеристики, които ще използваме.

# In[51]:


df.columns


# Не сме готови все още да използваме модел, тъй като имаме категориини променливи. Освен тях нека разгледаме и няколко числови променливи като например: 'first_player_id'. Стойностите са числа, но категорията не е линейна - тоест ако имаме играч с уникален номер 1 и играч срещу уникален номер 2, то по никакъв начин втория играч не е 2 пъти по различен от втория. Тези характеристики трябва да бъдат кодирани както категориините. Създаваме за всяка възможна стойност от тях - нова характеристика, където само мачовете имащи тази характеристика стават 1, а останалите 0.

# In[52]:


one_hot = pd.get_dummies(df['surface'], prefix=['surface'])
df = df.drop(['surface'],axis = 1)
df = df.join(one_hot)
one_hot = pd.get_dummies(df['tourney_name'], prefix=['tourney_name'])
df = df.drop(['tourney_name'],axis = 1)
df = df.join(one_hot)
one_hot = pd.get_dummies(df['tourney_level'], prefix =['tourney_level'])
df = df.drop(['tourney_level'],axis = 1)
df = df.join(one_hot)
one_hot = pd.get_dummies(df['first_player_id'], prefix =['first_player_id'])
df = df.drop(['first_player_id'],axis = 1)
df = df.join(one_hot)
one_hot = pd.get_dummies(df['first_player_hand'], prefix =['first_player_hand'])
df = df.drop(['first_player_hand'],axis = 1)
df = df.join(one_hot)
one_hot = pd.get_dummies(df['second_player_id'],prefix=['second_player_id'])
df = df.drop(['second_player_id'],axis = 1)
df = df.join(one_hot)
one_hot = pd.get_dummies(df['second_player_hand'], prefix = ['second_player_hand'])
df = df.drop(['second_player_hand'],axis = 1)
df = df.join(one_hot)
one_hot = pd.get_dummies(df['round'], prefix = ['round'])
df = df.drop(['round'],axis = 1)
df = df.join(one_hot)


# Ето така изглеждат част от характеристиките

# In[53]:


df.columns


# Размерността вече е доста по-голяма.

# In[54]:


df.shape


# Остана да отделим колоната с резултата от мача и сме готови да натренираме модел.

# In[55]:


y = df['outcome']
df_X = df.drop(columns='outcome')


# Разделяме данните на 3 части - тренировъчна, валидационна и тестова съответно 60%,20%,20% от всичко. 

# In[56]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


# Първо ще натренираме логистична регресия, ще изследваме параметрите с валидационното множество и накрая ще видим резултата от тестовото множество.

# Използваме готов модел от sklearn библиотеката.

# In[57]:


from sklearn.linear_model import LogisticRegression
for c in [0.001, 0.003, 0.006, 0.1, 0.2, 0.4, 1.0, 3.0, 6.0]:
    model = LogisticRegression(C=c, solver = 'liblinear', random_state=7777)
    model.fit(X_train, y_train)
    print("C={:f} Тренировъчни: {:f} Валидационни: {:f}".format(c, model.score(X_train, y_train), model.score(X_val, y_val)))


# In[77]:


model = LogisticRegression(C= 3, solver = 'liblinear', random_state=7777)
model.fit(X_train,y_train)
print("Тестови: ", model.score(X_test,y_test))


# Резултата не е висок, но имайки се предвид, че най-високия до сега е около 85%, то не сме се справили чак толкова зле. Нека сега разгледаме какво е научил модела

# In[78]:


explain = pd.DataFrame(zip(X_train.columns, np.transpose(model.coef_)), columns=['features', 'coef'])


# Първо ще покажем 10-те най важни критерия за да имаме победител втори играч според модела ни.

# In[79]:


explain['coef'] = explain['coef'].astype('float')
explain.nlargest(10,columns=['coef'])


# Изглежда най-важното е кой е играча. Нека видим кой е играча на първо място.

# In[80]:


df2 = pd.read_csv('data/matches/all_matches.csv', low_memory=False)
df2[df2['winner_id']== 101736]['winner_name'].value_counts()


# А за победа на първия играч?

# In[81]:


explain.nsmallest(10,columns=['coef'])


# In[96]:


df2[df2['winner_id']== 103720]['winner_name'].value_counts()


# Отново кой играе е в основата на нашето предсказване. Лейтън Хюи е бил номер 1 за доста дълго време и има страхотна статистика според сайта на АТП, което е нормално неговия номер да е с такъв коефициент. Естествено заради случайността в разпределянето на играч първи и играч втори спрямо това дали е победил или не, няма как да очакваме еднакви играчи в двете таблици с коефициентите, тъй като се е случило някой добър играч да е бил "домакин" в повечето от случите.

# За да придобием още малко информация за самия модел ще покажем матрица на объркванията

# In[83]:


from sklearn.metrics import plot_confusion_matrix
disp = plot_confusion_matrix(model, X_test, y_test, normalize = 'true', cmap=plt.cm.Blues)
plt.show()


# In[84]:


from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
precision = precision_score(y_test,model.predict(X_test), average='macro')
recall = recall_score(y_test, model.predict(X_test), average='macro')
F1score = 2*(precision*recall)/(precision+recall)
print('Precision Score: ',precision)
print('Recall Score: ', recall)
print('F1 Score: ', F1score)


# Нека опитаме с алгоритъма "Random Forest".

# In[66]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [5,50,100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [3,5,8],
    'criterion' :['gini', 'entropy']
}


CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
CV_rfc.best_params_


# In[85]:


from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=42, max_depth = 8, n_estimators = 50).fit(X_train, y_train)
print("Тренировъчен:", model2.score(X_train, y_train))
print("Валидационен: ", model2.score(X_val, y_val))
print('Тестов: ', model2.score(X_test,y_test))


# След като намерихме най-добрите параметри успяхме да вдигнем резултата до 0.67. С така манипулираните данни и тези два алгоритъма ориентировачния резултат които можем да получим е 0.70. Имайки предвид, че сферата е доста комплексна е сравнително добър резултат.

# ## 4. Моделът в реални условия

# Всички тези теоритични оценки за точността на модела са хубави, но всеки би се запитал едно нещо - достатъчно добър ли е този модел за да ни донесе приходи ако залагаме на базата на негова преценка. Сега ще разберем.

# Нека вземем 100 мача през 2020 година с равновероятно разпределение и заложим на всеки от тях с 1 лев. Накрая ще видим колко пари сме изкарали.

# In[86]:


betSample = X_test[X_test['tourney_year'] == 2020].sample(n=100, random_state = 7777)


# In[87]:


sampleOutcome = (y_test.loc[betSample.index])


# In[88]:


pd.set_option('display.max_rows', None)


# Ръчно ще проверим коефициентите за победителя във всеки мач благодарение на сайта: https://www.oddsportal.com

# In[89]:


coefficients = [1.27,1.37,1.47,3.19,1.57,1.04,3.41,15.90,1.49,1.05,
                1.25,1.06,1.24,3.10,1.38,1.64,1.56,2.08,13.84,1.42,
                1.45,1.33,1.67,2.79,1.37,3.05,1.75,2.06,2.97,1.53,
                1.15,2.56,1.36,2.42,1.37,1.02,1.37,2.86,2.20,1.78,
                1.63,1.19,1.23,1,2.36,2.36,1.37,1.42,2.66,2.02,
                1.31,1.20,1.62,1.62,1.39,2.29,2.22,1.12,1.37,1.80,
                1.05,1.29,1.54,1.27,2.97,1.24,1.52,1.45,1.50,1.17,
                1.48,1.21,1.16,3.02,1.23,1.70,2.14,2.20,2.09,2.36,
                3.20,1.41,1.70,2.44,3.30,1.27,1.13,2.46,1.25,1.43,
                1.69,1.20,1.91,1.02,3.95,1.01,1,1.30,1.63,1.85]


# In[90]:


df2.loc[betSample.index][['tourney_name','winner_name', 'loser_name']].head(100)


# In[91]:


sampleResult = [int(e1 == e2) for (e1,e2) in zip(sampleOutcome,model2.predict(betSample))]


# Нека първо видим точността за точно тези 10 мача.

# In[92]:


print("Извадкова точност: ",sum(sampleResult)/len(sampleResult))


# А сега и крайния резултат от залагането

# In[93]:


vector1 = np.array(sampleResult)
vector2 = np.array(coefficients)
vector3 = np.dot(vector1,vector2)


# In[94]:


print("Събрани пари: ",vector3)
print("Заложени пари: ", len(sampleResult))
print("Чиста печалба: ", vector3 - len(sampleResult))


# Изглежда, че за точно тези мачове, модела е достатъчно добър дори ни донесе и някаква печалба. Естествено, че няма всеки път резултата да е такъв и най-вероятно самата извадка е такава, но със сигурност ще опитаме съшия експеримент с мачовте от Откритото първенсто на Австралия другата седмица. 

# ## 5. Заключение

# Използвахме доста обемни и богати данни от тенис мачове за да изследваме този спорт. Опитахме се да обясним някои зависимости продиктувани от данните според нашите виждания - естествено може за някой неща да не сме били прави или да има такива, които не сме открили, но въпреки това направихме частичен анализ на характеристиките. Преди да хвърлим всички данни в модела ги почистихме и дори модифицирахме с идея за по-добър резултат. Сравнихме два класификационни алгоритъма, за които не просто казахме точността, а погледнахме какво точно са научили алгоритмите и дали правят разумна класификация. За да не оставим нещата единствено теоритични приложихме проблем от реалния свят и видяхме резултатите за мачове доста близо до настоящето. Моделът може да се подобри като манипулираме още повече данните - например може да добавим за всеки мач и информация за победите и загубите на играчите до момента на мача. Също така можем да използваме по евристично залагане - не на всеки мач по 1 лев, а в зависимост от това колко сигурен е модела в предсказанието.
